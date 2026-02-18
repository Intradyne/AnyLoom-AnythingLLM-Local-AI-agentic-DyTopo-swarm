// handler.js — smart-web-reader skill for AnythingLLM
// Fetches a web page and extracts clean readable content as markdown.
// Uses a three-tier extraction pipeline with graceful degradation.
// Returns a JSON envelope string. MUST always return a string.

module.exports.runtime = {
  handler: async function ({ url = "" }) {
    try {
      const MAX_OUTPUT = 8000;

      // --- Input validation ---
      const targetUrl = String(url).trim();

      if (!targetUrl) {
        return JSON.stringify({
          status: "error",
          source: "smart-web-reader",
          error:
            "No URL provided. Please supply a full URL starting with http:// or https://.",
        });
      }

      if (
        !targetUrl.startsWith("http://") &&
        !targetUrl.startsWith("https://")
      ) {
        return JSON.stringify({
          status: "error",
          source: "smart-web-reader",
          error: `Invalid URL: "${targetUrl}". URL must start with http:// or https://.`,
        });
      }

      // --- Strip fabricated Wikipedia anchor fragments ---
      let fetchUrl = targetUrl;
      if (targetUrl.includes("wikipedia.org") && targetUrl.includes("#")) {
        const fragment = targetUrl.split("#")[1];
        fetchUrl = targetUrl.split("#")[0];
        this.introspect(
          `Stripped anchor fragment "#${fragment}" — fetching base page instead.`
        );
      }

      // --- Fetch the page ---
      this.introspect(`Fetching web page: ${fetchUrl}`);

      const response = await fetch(fetchUrl, {
        method: "GET",
        headers: {
          "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
          Accept:
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
          "Accept-Language": "en-US,en;q=0.9",
        },
        signal: AbortSignal.timeout(15000),
      });

      if (!response.ok) {
        const statusCode = response.status;
        let guidance = `HTTP ${statusCode}: `;

        if (statusCode === 403) {
          guidance +=
            "Access forbidden. The site blocks automated requests. Try a different URL or use the built-in web-scraper.";
        } else if (statusCode === 404) {
          guidance +=
            "Page not found. This URL does not exist. Do NOT guess or fabricate alternative URLs. Instead, tell the user the page was not found, or use a web-search tool to find the correct URL.";
        } else if (statusCode === 429) {
          guidance +=
            "Too many requests. Wait a moment and try again.";
        } else if (statusCode >= 500) {
          guidance +=
            "Server error on the target site. Try again later.";
        } else {
          guidance += `${response.statusText}. Try a different URL or use the built-in web-scraper.`;
        }

        this.logger(`smart-web-reader fetch error: ${guidance}`);
        return JSON.stringify({
          status: "error",
          source: targetUrl,
          error: guidance,
        });
      }

      const html = await response.text();

      if (!html || html.trim().length === 0) {
        return JSON.stringify({
          status: "error",
          source: targetUrl,
          error:
            "Page returned empty content. The site may require JavaScript rendering. Try the built-in web-scraper instead.",
        });
      }

      this.introspect("Extracting readable content...");

      let title = "";
      let content = "";
      let extractionMethod = "";

      // ---------------------------------------------------------
      // Tier 1: Defuddle + JSDOM  (best quality)
      // ---------------------------------------------------------
      try {
        // Use root CJS export (defuddle/node is ESM-only, incompatible with Node 18 require())
        const Defuddle = require("defuddle");
        const { JSDOM } = require("jsdom");
        const TurndownService = require("turndown");

        const dom = new JSDOM(html, { url: fetchUrl });
        const d = new Defuddle(dom.window.document);
        // Suppress noisy CSS selector errors (e.g. MathJax selectors on Wikipedia)
        const _origErr = console.error;
        console.error = () => {};
        let result;
        try { result = d.parse(); } finally { console.error = _origErr; }

        if (result && result.content && result.content.trim().length > 0) {
          title = result.title || "";
          // Root export returns HTML; convert to markdown via Turndown
          const turndown = new TurndownService({
            headingStyle: "atx",
            codeBlockStyle: "fenced",
          });
          content = turndown.turndown(result.content).trim();
          extractionMethod = "defuddle";
          this.logger("smart-web-reader: extraction via Defuddle succeeded");
        }
      } catch (tier1Err) {
        this.logger(
          `smart-web-reader: Defuddle unavailable or failed: ${tier1Err.message}`
        );
      }

      // ---------------------------------------------------------
      // Tier 2: Readability + Turndown  (good fallback)
      // ---------------------------------------------------------
      if (!content) {
        try {
          const { Readability } = require("@mozilla/readability");
          const { JSDOM } = require("jsdom");
          const TurndownService = require("turndown");

          const dom = new JSDOM(html, { url: fetchUrl });
          const reader = new Readability(dom.window.document);
          const article = reader.parse();

          if (article && article.content && article.content.trim().length > 0) {
            const turndown = new TurndownService({
              headingStyle: "atx",
              codeBlockStyle: "fenced",
            });
            title = article.title || "";
            content = turndown.turndown(article.content).trim();
            extractionMethod = "readability+turndown";
            this.logger(
              "smart-web-reader: extraction via Readability+Turndown succeeded"
            );
          }
        } catch (tier2Err) {
          this.logger(
            `smart-web-reader: Readability unavailable or failed: ${tier2Err.message}`
          );
        }
      }

      // ---------------------------------------------------------
      // Tier 3: Regex-based HTML stripping  (last resort)
      // ---------------------------------------------------------
      if (!content) {
        try {
          // Extract title from <title> tag
          const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
          title = titleMatch ? titleMatch[1].trim() : "";

          let cleaned = html;
          // Remove script and style blocks
          cleaned = cleaned.replace(
            /<script[\s\S]*?<\/script>/gi,
            ""
          );
          cleaned = cleaned.replace(
            /<style[\s\S]*?<\/style>/gi,
            ""
          );
          // Remove HTML comments
          cleaned = cleaned.replace(/<!--[\s\S]*?-->/g, "");
          // Remove all HTML tags
          cleaned = cleaned.replace(/<[^>]+>/g, " ");
          // Decode common HTML entities
          cleaned = cleaned
            .replace(/&amp;/g, "&")
            .replace(/&lt;/g, "<")
            .replace(/&gt;/g, ">")
            .replace(/&quot;/g, '"')
            .replace(/&#39;/g, "'")
            .replace(/&nbsp;/g, " ");
          // Normalize whitespace
          cleaned = cleaned.replace(/[ \t]+/g, " ");
          cleaned = cleaned.replace(/\n{3,}/g, "\n\n");
          content = cleaned.trim();
          extractionMethod = "regex-fallback";
          this.logger(
            "smart-web-reader: extraction via regex fallback"
          );
        } catch (tier3Err) {
          this.logger(
            `smart-web-reader: regex fallback failed: ${tier3Err.message}`
          );
        }
      }

      if (!content) {
        return JSON.stringify({
          status: "error",
          source: targetUrl,
          error:
            "Failed to extract readable content from the page. The site may rely entirely on JavaScript rendering. Try the built-in web-scraper instead.",
        });
      }

      // --- Truncate to max length ---
      if (content.length > MAX_OUTPUT) {
        content = content.substring(0, MAX_OUTPUT) + "\n\n[Content truncated at 8,000 characters]";
      }

      // Sanitize title
      title = title.replace(/\s+/g, " ").trim();
      if (title.length > 200) {
        title = title.substring(0, 200) + "...";
      }

      this.introspect(
        `Extracted ${content.length} characters using ${extractionMethod}${title ? `: "${title}"` : ""}`
      );

      // --- Content-relevance hint ---
      const wordCount = content.split(/\s+/).length;
      if (wordCount < 50) {
        content = "[Page contained very little readable text. This may be the wrong page for your query.]";
      }

      return JSON.stringify({
        status: "success",
        source: targetUrl,
        data: {
          title: title,
          content: content,
          url: targetUrl,
          word_count: wordCount,
          extraction_method: extractionMethod,
        },
      });
    } catch (err) {
      const message = err && err.message ? err.message : String(err);
      this.logger(`smart-web-reader unhandled error: ${message}`);

      let guidance = message;
      if (message.includes("TimeoutError") || message.includes("abort")) {
        guidance =
          "Request timed out after 15 seconds. The site may be slow or unreachable. Try again or use a different URL.";
      } else if (message.includes("fetch") || message.includes("network")) {
        guidance =
          "Network error. Check container internet access and DNS resolution.";
      }

      return JSON.stringify({
        status: "error",
        source: "smart-web-reader",
        error:
          "Failed to fetch URL. " +
          guidance +
          " The site may block automated requests. Try a different URL or use the built-in web-scraper.",
      });
    }
  },
};
