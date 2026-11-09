import Link from 'next/link';
import { Button } from '@/components/ui/button';
export default function NotFound() {
  return (
    <div className="container py-24">
      <div className="eyebrow">PAGE NOT FOUND</div>
      <h1>Let’s get you back to the lab.</h1>
      <p className="mt-4 mb-6">This page does not exist. The models and data are a click away.</p>
      <Button asChild>
        <Link href="/">Return to FlowLens</Link>
      </Button>
    </div>
  );
}
