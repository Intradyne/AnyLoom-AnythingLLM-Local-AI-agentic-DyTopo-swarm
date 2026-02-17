// handler.js — asset-price skill for AnythingLLM
// Fetches current market prices for stocks, crypto, commodities, indices, ETFs.
// Uses yahoo-finance2 v3 for broad asset coverage. No API key needed.
// Returns a JSON envelope string. MUST always return a string.

const ALIASES = {
  // Precious metals (futures)
  gold: "GC=F",
  silver: "SI=F",
  platinum: "PL=F",
  palladium: "PA=F",
  copper: "HG=F",
  // Crypto
  bitcoin: "BTC-USD",
  btc: "BTC-USD",
  ethereum: "ETH-USD",
  eth: "ETH-USD",
  solana: "SOL-USD",
  sol: "SOL-USD",
  dogecoin: "DOGE-USD",
  doge: "DOGE-USD",
  xrp: "XRP-USD",
  ripple: "XRP-USD",
  // Indices
  "s&p 500": "^GSPC",
  "s&p500": "^GSPC",
  sp500: "^GSPC",
  nasdaq: "^IXIC",
  "dow jones": "^DJI",
  dow: "^DJI",
  russell: "^RUT",
  "russell 2000": "^RUT",
  vix: "^VIX",
  // Energy
  oil: "CL=F",
  "crude oil": "CL=F",
  "natural gas": "NG=F",
  // Currencies
  euro: "EURUSD=X",
  yen: "JPY=X",
  pound: "GBPUSD=X",
};

module.exports.runtime = {
  handler: async function ({ query = "" }) {
    try {
      const input = String(query).trim();

      if (!input) {
        return JSON.stringify({
          status: "error",
          source: "asset-price",
          error:
            "No query provided. Give a ticker symbol (AAPL, BTC-USD) or common name (silver, bitcoin, S&P 500).",
        });
      }

      // Resolve alias or use raw input as symbol
      const symbol = ALIASES[input.toLowerCase()] || input.toUpperCase();

      this.introspect(`Looking up price for: ${symbol}`);

      // --- Try yahoo-finance2 v3 ---
      let yf;
      try {
        const YahooFinance = require("yahoo-finance2").default;
        yf = new YahooFinance({ suppressNotices: ["yahooSurvey"] });
      } catch (e) {
        this.logger("asset-price: yahoo-finance2 not available: " + e.message);
        return JSON.stringify({
          status: "error",
          source: "asset-price",
          error:
            "yahoo-finance2 package not installed. Run: docker exec anyloom-anythingllm npm install --prefix /app/server --legacy-peer-deps yahoo-finance2",
        });
      }

      let quote;
      try {
        quote = await yf.quote(symbol);
      } catch (e) {
        // If direct symbol fails, try searching
        this.logger(`asset-price: quote(${symbol}) failed: ${e.message}`);
        try {
          this.introspect(`Symbol "${symbol}" not found directly, searching...`);
          const search = await yf.search(input, { newsCount: 0 });
          if (search.quotes && search.quotes.length > 0) {
            const best = search.quotes[0];
            quote = await yf.quote(best.symbol);
          }
        } catch (searchErr) {
          this.logger("asset-price: search fallback failed: " + searchErr.message);
        }
      }

      if (!quote || !quote.symbol) {
        return JSON.stringify({
          status: "error",
          source: "asset-price",
          error: `Could not find "${input}". Try a specific ticker symbol (e.g., AAPL, BTC-USD, GC=F, ^GSPC).`,
        });
      }

      const price = quote.regularMarketPrice ?? null;
      const change = quote.regularMarketChange ?? null;
      const changePct = quote.regularMarketChangePercent ?? null;
      const currency = quote.currency || "USD";

      this.introspect(
        `${quote.shortName || quote.symbol}: ${currency} ${price != null ? price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "N/A"}` +
          (change != null ? ` (${change >= 0 ? "+" : ""}${change.toFixed(2)}, ${changePct != null ? changePct.toFixed(2) : "?"}%)` : "")
      );

      return JSON.stringify({
        status: "success",
        source: "yahoo-finance",
        data: {
          symbol: quote.symbol,
          name: quote.shortName || quote.longName || quote.symbol,
          type: quote.quoteType || "unknown",
          price: price,
          currency: currency,
          change: change,
          change_percent: changePct,
          day_high: quote.regularMarketDayHigh ?? null,
          day_low: quote.regularMarketDayLow ?? null,
          previous_close: quote.regularMarketPreviousClose ?? null,
          volume: quote.regularMarketVolume ?? null,
          market_cap: quote.marketCap ?? null,
          exchange: quote.exchange || null,
          market_state: quote.marketState || null,
        },
      });
    } catch (err) {
      const message = err && err.message ? err.message : String(err);
      this.logger(`asset-price unhandled error: ${message}`);

      let guidance = message;
      if (message.includes("TimeoutError") || message.includes("abort")) {
        guidance =
          "Request timed out. Yahoo Finance may be slow or unreachable. Try again shortly.";
      } else if (message.includes("fetch") || message.includes("network")) {
        guidance =
          "Network error reaching Yahoo Finance. Check container internet access.";
      }

      return JSON.stringify({
        status: "error",
        source: "asset-price",
        error: guidance,
      });
    }
  },
};
