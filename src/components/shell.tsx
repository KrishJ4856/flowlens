'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ThemeProvider, useTheme } from 'next-themes';
import { Activity, ArrowUpRight, CodeXml as Github, Menu, Moon, Sun, X } from 'lucide-react';
import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { TooltipProvider } from '@/components/ui/tooltip';
import { repository } from '@/lib/features';
const navigation = [
  ['Home', '/'],
  ['Playground', '/playground/'],
  ['Training', '/training/'],
  ['Testing', '/testing/'],
  ['Analysis', '/analysis/'],
  ['About', '/project/'],
];
function Navigation({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [open, setOpen] = useState(false);
  useEffect(() => setMounted(true), []);
  return (
    <>
      <header className="site-header">
        <div className="header-inner">
          <Link href="/" className="brand" aria-label="FlowLens home">
            <span className="brand-mark">
              <Activity size={22} />
            </span>
            <span>FlowLens</span>
          </Link>
          <nav className="desktop-nav" aria-label="Main navigation">
            {navigation.map(([label, path]) => (
              <Link
                key={path}
                href={path}
                className={pathname === path ? 'nav-link active' : 'nav-link'}
              >
                {label}
              </Link>
            ))}
          </nav>
          <div className="header-actions">
            <Button
              variant="ghost"
              size="icon"
              aria-label={
                mounted && resolvedTheme === 'dark'
                  ? 'Switch to light theme'
                  : 'Switch to dark theme'
              }
              onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
            >
              {mounted && resolvedTheme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </Button>
            <a
              className="github-icon"
              href={repository}
              target="_blank"
              rel="noreferrer"
              aria-label="View GitHub repository"
            >
              <Github size={19} />
            </a>
            <Button
              className="mobile-menu"
              variant="ghost"
              size="icon"
              aria-label={open ? 'Close navigation' : 'Open navigation'}
              onClick={() => setOpen(!open)}
            >
              {open ? <X /> : <Menu />}
            </Button>
          </div>
        </div>
        {open && (
          <nav className="mobile-nav" aria-label="Mobile navigation">
            {navigation.map(([label, path]) => (
              <Link key={path} href={path} onClick={() => setOpen(false)}>
                {label}
              </Link>
            ))}
          </nav>
        )}
      </header>
      <main id="main-content">{children}</main>
      <footer className="site-footer">
        <span>UNSW-NB15 · Network traffic dataset</span>
        <div>
          <Link href="/project/">About the project</Link>
          <a href={repository} target="_blank" rel="noreferrer">
            GitHub <ArrowUpRight size={13} />
          </a>
        </div>
      </footer>
    </>
  );
}
export function Shell({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
      <TooltipProvider delayDuration={150}>
        <a href="#main-content" className="skip-link">
          Skip to content
        </a>
        <Navigation>{children}</Navigation>
      </TooltipProvider>
    </ThemeProvider>
  );
}
