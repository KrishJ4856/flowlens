**Understanding the data without ML jargon**

A **network** lets computers communicate. A **packet** is a small piece of data sent through that network. A **flow** summarizes a communication between two endpoints. Each row in this project is a record of measurements about a flow. We are classifying those measurements, rather than reading the text of someone's messages.

The **source** is the sending endpoint; the **destination** is the receiving endpoint. In many column names, `s` refers to source and `d` to destination. A **byte** is a unit of data size. A **protocol** is a set of communication rules; TCP and UDP are examples. A **service** indicates the kind of communication, such as HTTP or DNS. You do not need to memorize every feature to explain the project.

**The three kinds of columns**

| Kind | Examples | How we use it |
| --- | --- | --- |
| Measurements | dur, spkts, sbytes, proto | Inputs that the model uses to learn and predict |
| Answers | label, attack_cat | Used for teaching and checking; never given as prediction inputs |
| Local row number | id | Used to trace records in output; excluded from learning |

`label=0` means Normal and `label=1` means Attack. `attack_cat` names the category. It contains Normal, Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode and Worms. The primary model only answers Normal/Attack. The separately trained extension tries to name a known category.

**Useful measurements to recognise**

| Column or family | Plain-language description |
| --- | --- |
| dur | How long the recorded communication lasted |
| proto | Communication protocol |
| service | Service recorded for the communication; '-' is retained as a category |
| state | Recorded communication state |
| spkts, dpkts | Packet counts in the two directions |
| sbytes, dbytes | Amounts of data sent in the two directions |
| rate | The partition's precomputed flow-rate measurement; do not infer a new formula from the name |
| sttl, dttl | Packet time-to-live values in the two directions; these are packet-header settings, not the same as flow duration |
| sload, dload | Data transfer rates, described in the source dictionary as bits per second |
| sloss, dloss | Recorded retransmitted or dropped packet counts |
| sinpkt, dinpkt | Time between packet arrivals in the two directions |
| sjit, djit | Variation in packet timing in the two directions |
| swin, dwin | TCP window advertisement values |
| stcpb, dtcpb | TCP base sequence numbers |
| tcprtt, synack, ackdat | Measurements related to setting up a TCP connection |
| smean, dmean | Mean packet sizes in the two directions |
| trans_depth, response_body_len | Measurements related to HTTP requests and responses |
| ct_* | Precomputed counts describing related connections or activity patterns |
| is_ftp_login | Dataset indicator related to FTP login activity |
| is_sm_ips_ports | Dataset indicator for matching source/destination address and port values |

The original wider-dataset dictionary is included as data/raw/NUSW-NB15_features.csv, using the mirror's original filename. Some names differ from the reduced partition: smeansz/dmeansz correspond to smean/dmean, and res_bdy_len corresponds to response_body_len. The wider dictionary includes raw IP/port/time fields that are absent from this reduced partition. The exact model-required columns are listed in data/model_input_schema.json and data/column_schema.csv.

**What preprocessing does here**

The model needs consistent numeric inputs. One-hot encoding turns a category into separate indicator columns: for example, whether protocol is TCP or UDP. Numeric measurements are kept numeric. Rules learned from training handle missing values. The same saved preparation is applied during prediction, so the meaning and order of model inputs stay consistent.

**What “learning” means**

The algorithm searches for useful relationships between measurements and the supplied answers. The trained trees capture combinations of those measurements that help distinguish the labelled examples. A single high packet count does not automatically prove an attack; the model is using patterns learned from this particular dataset.

**What the dataset cannot establish on its own**

These are collected and labelled benchmark records. Successful classification here does not establish that the same model works equally well on another network or a new attack type. Some inputs depend on a flow or a history window being available. A future live system needs compatible measurement extraction and its own validation.

Source: the included feature dictionary and [the official UNSW-NB15 description](https://research.unsw.edu.au/projects/unsw-nb15-dataset). The exact original 42 measurement names in the partition, rather than the wider dataset's total feature count, define the input contract for this project.
