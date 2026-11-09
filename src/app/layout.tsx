import type { Metadata } from 'next';
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
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
