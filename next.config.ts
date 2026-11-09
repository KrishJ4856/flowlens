import type { NextConfig } from 'next';
const nextConfig: NextConfig = {
  output: 'export',
  distDir: process.env.NODE_ENV === 'development' ? '.next-dev' : '.next',
  outputFileTracingRoot: process.cwd(),
  trailingSlash: true,
  images: { unoptimized: true },
  experimental: { cpus: 2 },
};
export default nextConfig;
