// @ts-check
const { defineConfig, devices } = require('@playwright/test');

/**
 * Three viewports we explicitly support: a real phone, a tablet, and
 * a laptop-class desktop. Each runs the same suite — a regression in
 * one form factor is a real bug.
 */
const baseURL = process.env.TEST_BASE_URL || 'http://127.0.0.1:1314';

module.exports = defineConfig({
  testDir: './ui',
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',

  use: {
    baseURL,
    actionTimeout: 8_000,
    trace: 'retain-on-failure',
  },

  /* Start Hugo once for the whole run. We use a non-default port (1314)
     so test runs don't clash with `make preview`. Override the binary
     path with HUGO_BIN if hugo isn't on PATH. */
  webServer: process.env.TEST_BASE_URL
    ? undefined
    : {
        command: `${process.env.HUGO_BIN || 'hugo'} server --port 1314 --bind 127.0.0.1 --baseURL http://127.0.0.1:1314/ --appendPort=false --quiet`,
        url: 'http://127.0.0.1:1314/',
        reuseExistingServer: !process.env.CI,
        timeout: 60_000,
        cwd: '..',
      },

  /* All three projects use Chromium so they work in any environment that
     has just `chromium` installed (no webkit dependency). We override
     the user-agent and viewport so we still exercise the responsive
     CSS as if on a real phone / tablet. */
  projects: [
    {
      name: 'mobile',
      use: {
        ...devices['Pixel 5'],
        // Force chromium engine even on machines where Pixel 5 default would be webkit.
        browserName: 'chromium',
        launchOptions: process.env.CHROMIUM_BIN ? { executablePath: process.env.CHROMIUM_BIN } : undefined,
      },
    },
    {
      name: 'tablet',
      use: {
        browserName: 'chromium',
        launchOptions: process.env.CHROMIUM_BIN ? { executablePath: process.env.CHROMIUM_BIN } : undefined,
        viewport: { width: 820, height: 1180 },
        deviceScaleFactor: 2,
        isMobile: true,
        hasTouch: true,
      },
    },
    {
      name: 'desktop',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: process.env.CHROMIUM_BIN ? { executablePath: process.env.CHROMIUM_BIN } : undefined,
        viewport: { width: 1400, height: 900 },
      },
    },
  ],
});
