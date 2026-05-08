import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Only run our own tests/ tree; the Wave 0 reference repos in
    // _external/ and any built output in dist/ ship their own *.test.js
    // files which would otherwise be discovered (and fail because
    // they expect a different repo layout).
    include: ["tests/**/*.test.ts"],
    exclude: [
      "node_modules/**",
      "dist/**",
      "_external/**",
      ".git/**",
    ],
    globals: false,
    environment: "node",
    testTimeout: 30_000,
  },
});
