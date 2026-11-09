import type { Metadata } from 'next';
import localFont from 'next/font/local';
const bricolage = localFont({
  src: [
    { path: './fonts/bricolage-400.ttf', weight: '400', style: 'normal' },
    { path: './fonts/bricolage-700.ttf', weight: '700', style: 'normal' },
  ],
  variable: '--font-bricolage',
  display: 'swap',
});
import './globals.css';
import { Shell } from '@/components/shell';
export const metadata: Metadata = {
  title: { default: 'FlowLens — Network Traffic Lab', template: '%s · FlowLens' },
  description:
    'Understand network intrusion detection. Explore real data, compare two trained models, and learn what their results mean.',
};
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={bricolage.variable}>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
