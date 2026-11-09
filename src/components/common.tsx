import { ArrowUpRight, Info } from 'lucide-react';
import Link from 'next/link';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
export function PageHeading({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string;
  title: string;
  description: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {children}
    </div>
  );
}
export function Explain({ children }: { children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button type="button" className="info-button" aria-label="Explain this term">
          <Info size={14} />
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-72 text-sm leading-relaxed">{children}</TooltipContent>
    </Tooltip>
  );
}
export function Note({ children }: { children: React.ReactNode }) {
  return (
    <div className="note">
      <Info size={17} />
      <div>{children}</div>
    </div>
  );
}
export function Step({
  n,
  title,
  children,
}: {
  n: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="learning-step">
      <span>{n}</span>
      <div>
        <h3>{title}</h3>
        <p>{children}</p>
      </div>
    </div>
  );
}
export function SectionTitle({
  label,
  title,
  href,
}: {
  label?: string;
  title: string;
  href?: string;
}) {
  return (
    <div className="section-title">
      <div>
        {label && <div className="eyebrow">{label}</div>}
        <h2>{title}</h2>
      </div>
      {href && (
        <Link href={href}>
          Take a closer look <ArrowUpRight size={15} />
        </Link>
      )}
    </div>
  );
}
