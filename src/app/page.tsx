import Link from 'next/link';
import {
  ArrowUpRight,
  FlaskConical,
  Database,
  ClipboardCheck,
  ChartNoAxesCombined,
  ArrowRight,
} from 'lucide-react';
const cards = [
  {
    title: 'Playground',
    text: 'Change a connection. Compare two predictions.',
    href: '/playground/',
    icon: FlaskConical,
    meta: 'XGBoost + Random Forest',
    number: '01',
  },
  {
    title: 'Training data',
    text: 'The examples the models learned from.',
    href: '/training/',
    icon: Database,
    meta: '175,341 records',
    number: '02',
  },
  {
    title: 'Testing data',
    text: 'The records used to check their answers.',
    href: '/testing/',
    icon: ClipboardCheck,
    meta: '82,332 records',
    number: '03',
  },
  {
    title: 'Analysis',
    text: 'What they caught. What they missed.',
    href: '/analysis/',
    icon: ChartNoAxesCombined,
    meta: 'Results & comparisons',
    number: '04',
  },
];
export default function Home() {
  return (
    <div className="container home-page">
      <section className="home-intro">
        <span className="eyebrow">A NETWORK TRAFFIC LAB</span>
        <h1>
          Normal traffic.
          <br />
          <span>Or an attack?</span>
        </h1>
        <div className="intro-side">
          <p>
            Explore real connections.
            <br />
            See what two trained models detect.
          </p>
          <Link href="/project/" className="text-link">
            How it works <ArrowRight size={15} />
          </Link>
        </div>
      </section>
      <div className="home-grid">
        {cards.map((card) => (
          <Link href={card.href} key={card.title} className="home-card">
            <div className="home-card-top">
              <span className="card-icon">
                <card.icon size={25} strokeWidth={1.5} />
              </span>
              <span className="card-number">{card.number}</span>
            </div>
            <div className="home-card-body">
              <div>
                <h2>{card.title}</h2>
                <p>{card.text}</p>
              </div>
              <ArrowUpRight size={22} strokeWidth={1.5} />
            </div>
            <span className="card-meta">{card.meta}</span>
          </Link>
        ))}
      </div>
      <div className="home-caption">
        <span className="status-dot" /> Real models. Recorded data.
        <span className="caption-right">UNSW-NB15 · Offline classification</span>
      </div>
    </div>
  );
}
