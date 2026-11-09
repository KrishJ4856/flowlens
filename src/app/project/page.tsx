import Link from 'next/link';
import {
  ArrowRight,
  ArrowUpRight,
  Database,
  Braces,
  Layers,
  FileChartColumn,
  FolderCode,
} from 'lucide-react';
import { PageHeading } from '@/components/common';
import { repository } from '@/lib/features';
const files = [
  ['Data', 'ml/data/raw/', 'Training and testing CSVs.', Database],
  ['Models', 'ml/models/', 'The two saved detectors.', Layers],
  ['ML code', 'ml/scripts/', 'Train, test and predict.', Braces],
  ['Results', 'ml/results/', 'Scores and individual predictions.', FileChartColumn],
  ['Website', 'src/', 'Pages, components and styles.', FolderCode],
  ['Presentation', 'docs/', 'Report, charts and speaking notes.', FileChartColumn],
] as const;
const questions = [
  [
    'What is a network flow?',
    'A record summarizing a communication between computers. It describes things like duration, packet counts and bytes transferred—not the text of messages.',
  ],
  [
    'What are features and labels?',
    'A feature is one measurement. Each flow has 42. A label is the known answer: Normal (0) or Attack (1). The answer columns and record ID never enter the model as inputs.',
  ],
  [
    'How do the two models differ?',
    'Random Forest averages predictions from 200 randomized trees. XGBoost combines 700 trees built progressively. They learn different patterns, so they can disagree.',
  ],
  [
    'What happens when I press Run?',
    'The browser applies the saved preprocessing, traverses the original fitted trees and compares each attack score with its saved cutoff: 0.50 for XGBoost, 0.55 for Random Forest. It does not retrain. Scores are not calibrated real-world risk percentages.',
  ],
  [
    'How was the experiment checked?',
    'Settings were selected using grouped validation inside training, then frozen before final testing. The browser produced the same Normal/Attack decisions as Python on all 82,332 test records. The repository includes the code and verification results.',
  ],
  [
    'What are the limitations?',
    'On the complete test, both models catch about 98% of attacks but wrongly flag around one in four normal flows. Some test patterns repeat training measurements; separate no-exact-match results are reported. Performance on a live network or new attack types has not been established.',
  ],
];
export default function Project() {
  return (
    <div className="container page-content">
      <PageHeading
        title="How it works."
        description="A model learns from examples. Then we check what it learned."
      />
      <div className="about-flow">
        <div>
          <span>01 / LEARN</span>
          <h2>Show examples.</h2>
          <p>175,341 flows, each with a known answer.</p>
        </div>
        <ArrowRight className="flow-arrow" size={20} />
        <div>
          <span>02 / CHECK</span>
          <h2>Take the test.</h2>
          <p>82,332 separate records. Count the mistakes.</p>
        </div>
        <ArrowRight className="flow-arrow" size={20} />
        <div>
          <span>03 / EXPLORE</span>
          <h2>Try a connection.</h2>
          <p>Change measurements. Compare both models.</p>
        </div>
      </div>
      <div className="about-columns">
        <section>
          <h2 className="section-heading">The basics</h2>
          <div className="question-list">
            {questions.map(([title, text]) => (
              <details key={title}>
                <summary>{title}</summary>
                <p>{text}</p>
              </details>
            ))}
          </div>
        </section>
        <section>
          <div className="section-title">
            <h2 className="section-heading">Inside the project</h2>
            <a href={repository} target="_blank" rel="noreferrer" className="text-link">
              GitHub <ArrowUpRight size={14} />
            </a>
          </div>
          <div className="asset-list">
            {files.map(([label, path, description, Icon]) => (
              <a
                href={`${repository}/tree/main/${path}`}
                key={path}
                target="_blank"
                rel="noreferrer"
              >
                <Icon size={19} strokeWidth={1.5} />
                <div>
                  <b>{label}</b>
                  <p>{description}</p>
                </div>
                <ArrowUpRight size={15} />
              </a>
            ))}
          </div>
          <a
            href="https://research.unsw.edu.au/projects/unsw-nb15-dataset"
            target="_blank"
            rel="noreferrer"
            className="source-credit"
          >
            Dataset: UNSW-NB15 <ArrowUpRight size={12} />
          </a>
        </section>
      </div>
      <Link className="bottom-cta" href="/playground/">
        Ready to try a record?{' '}
        <span>
          Open playground <ArrowRight size={16} />
        </span>
      </Link>
    </div>
  );
}
