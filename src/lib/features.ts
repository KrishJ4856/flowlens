export type Values = Record<string, string | number | null>;
export type Sample = {
  id: string;
  title: string;
  source: string;
  values: Values;
  truth: number | null;
  category: string;
  recordId: number;
};
export const groups = [
  {
    id: 'connection',
    title: 'Connection basics',
    description: 'The type of communication and how it ended.',
    fields: ['dur', 'proto', 'service', 'state', 'sttl', 'dttl', 'is_sm_ips_ports'],
  },
  {
    id: 'volume',
    title: 'Packets & data',
    description: 'How much data moved in each direction.',
    fields: [
      'spkts',
      'dpkts',
      'sbytes',
      'dbytes',
      'rate',
      'sload',
      'dload',
      'sloss',
      'dloss',
      'smean',
      'dmean',
    ],
  },
  {
    id: 'timing',
    title: 'Timing & TCP',
    description: 'Packet timing and details of setting up the connection.',
    fields: [
      'sinpkt',
      'dinpkt',
      'sjit',
      'djit',
      'swin',
      'dwin',
      'stcpb',
      'dtcpb',
      'tcprtt',
      'synack',
      'ackdat',
    ],
  },
  {
    id: 'activity',
    title: 'Related activity',
    description: 'Counts and indicators from surrounding connections and services.',
    fields: [
      'trans_depth',
      'response_body_len',
      'ct_srv_src',
      'ct_state_ttl',
      'ct_dst_ltm',
      'ct_src_dport_ltm',
      'ct_dst_sport_ltm',
      'ct_dst_src_ltm',
      'is_ftp_login',
      'ct_ftp_cmd',
      'ct_flw_http_mthd',
      'ct_src_ltm',
      'ct_srv_dst',
    ],
  },
];
export const details: Record<string, [string, string]> = {
  dur: ['Duration', 'How long this flow lasted, in seconds.'],
  proto: ['Protocol', 'The communication protocol, such as TCP or UDP.'],
  service: [
    'Service',
    'The recorded service, such as HTTP or DNS. A dash is a retained dataset category.',
  ],
  state: ['Connection state', 'The state recorded for this communication.'],
  sttl: [
    'Source TTL',
    'Source packet time-to-live: a packet-header value, not communication duration.',
  ],
  dttl: ['Destination TTL', 'Destination packet time-to-live value.'],
  is_sm_ips_ports: [
    'Matching endpoints',
    'Dataset indicator for matching source/destination address and port values.',
  ],
  spkts: ['Source packets', 'Number of packets sent from the source.'],
  dpkts: ['Destination packets', 'Number of packets sent in the reverse direction.'],
  sbytes: ['Source bytes', 'Amount of data sent from the source, in bytes.'],
  dbytes: ['Destination bytes', 'Amount of data sent in the reverse direction, in bytes.'],
  rate: [
    'Flow rate',
    'The rate measurement already calculated in the dataset. Editing packet counts does not recalculate this field.',
  ],
  sload: ['Source load', 'Source data transfer rate, in bits per second.'],
  dload: ['Destination load', 'Destination data transfer rate, in bits per second.'],
  sloss: ['Source loss', 'Recorded retransmitted or dropped source packets.'],
  dloss: ['Destination loss', 'Recorded retransmitted or dropped destination packets.'],
  smean: ['Mean source packet', 'Average source packet size.'],
  dmean: ['Mean destination packet', 'Average destination packet size.'],
  sinpkt: [
    'Source packet interval',
    'Time between source packet arrivals, recorded in milliseconds.',
  ],
  dinpkt: [
    'Destination packet interval',
    'Time between destination packet arrivals, recorded in milliseconds.',
  ],
  sjit: ['Source jitter', 'Variation in source packet timing, recorded in milliseconds.'],
  djit: ['Destination jitter', 'Variation in destination packet timing, recorded in milliseconds.'],
  swin: ['Source TCP window', 'Source TCP window advertisement value.'],
  dwin: ['Destination TCP window', 'Destination TCP window advertisement value.'],
  stcpb: ['Source sequence number', 'Source TCP base sequence number.'],
  dtcpb: ['Destination sequence number', 'Destination TCP base sequence number.'],
  tcprtt: ['TCP round-trip time', 'Measured time for the TCP connection setup, in seconds.'],
  synack: ['SYN to SYN-ACK', 'Connection-setup time between SYN and SYN-ACK, in seconds.'],
  ackdat: ['SYN-ACK to ACK', 'Connection-setup time between SYN-ACK and ACK, in seconds.'],
  trans_depth: ['HTTP depth', 'Recorded HTTP request/response transaction depth.'],
  response_body_len: ['Response body size', 'Size of the recorded HTTP response body.'],
  ct_srv_src: [
    'Source/service count',
    'Related recent connections with the same source and service.',
  ],
  ct_state_ttl: [
    'State/TTL count',
    'Dataset count based on connection state and TTL characteristics.',
  ],
  ct_dst_ltm: ['Destination count', 'Recent connections with the same destination.'],
  ct_src_dport_ltm: [
    'Source/destination-port count',
    'Recent connections sharing a source and destination port.',
  ],
  ct_dst_sport_ltm: [
    'Destination/source-port count',
    'Recent connections sharing a destination and source port.',
  ],
  ct_dst_src_ltm: ['Endpoint-pair count', 'Recent connections sharing a source/destination pair.'],
  is_ftp_login: ['FTP login indicator', 'Dataset flag related to FTP login activity.'],
  ct_ftp_cmd: ['FTP command count', 'Recorded count of FTP commands.'],
  ct_flw_http_mthd: ['HTTP method count', 'Recorded HTTP method activity count.'],
  ct_src_ltm: ['Source count', 'Recent connections from the same source.'],
  ct_srv_dst: [
    'Destination/service count',
    'Related recent connections with the same destination and service.',
  ],
};
export const repository = 'https://github.com/KrishJ4856/flowlens';
export async function readGzip(url: string) {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error('The data could not be loaded. Check your connection and retry.');
  if (!response.body || typeof DecompressionStream === 'undefined')
    throw new Error('Please use a recent browser that supports gzip decompression.');
  return new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).json();
}
export function formatNumber(n: number) {
  return new Intl.NumberFormat('en-IN').format(n);
}
