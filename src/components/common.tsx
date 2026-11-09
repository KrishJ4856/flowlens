import { Info } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
export function PageHeading({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        {description && <p>{description}</p>}
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
      <Info size={16} />
      <div>{children}</div>
    </div>
  );
}
