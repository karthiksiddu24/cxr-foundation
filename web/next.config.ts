import type { NextConfig } from "next";

const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
];

const config: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
  // The browser talks to the API through this same-origin proxy, so no CORS is needed in
  // production and the API URL stays a server-side setting.
  async rewrites() {
    const api = process.env.API_INTERNAL_URL ?? "http://localhost:8000";
    return [
      { source: "/api/v1/:path*", destination: `${api}/v1/:path*` },
      { source: "/api/ready", destination: `${api}/ready` },
      { source: "/api/health", destination: `${api}/health` },
    ];
  },
};

export default config;
