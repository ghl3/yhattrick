/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async redirects() {
    return [
      // the xG diagnostics live in the /models explainer
      { source: "/xg", destination: "/models", permanent: true },
    ];
  },
};

export default nextConfig;
