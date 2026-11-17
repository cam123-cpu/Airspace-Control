# Threat model: deployed counter-drone detection system

Version 0.1, 2026-11-16. Status: **draft for engineering review**. Sections marked [Legal review] need review by counsel who knows counter-UAS, surveillance and state corrections law; nothing here is legal advice.

## 1. System in scope

Customer: a state corrections department with 30 sites under one central contract.

| # | Component | Where | Notes |
|---|---|---|---|
| C1 | Radar | Tower, per site | Emits; needs FCC radiolocation licensing ([2020 interagency advisory](https://www.justice.gov/media/1084061/dl)) |
| C2 | RF detection and direction-finding nodes (2–4) | Roof and fence corners | Passive receive; captures control-link metadata [Legal review] |
| C3 | Remote ID receiver | Same as C2 | Wi-Fi/Bluetooth broadcast receiver; **unauthenticated input from the air** |
| C4 | Acoustic nodes | Fence line | Passive |
| C5 | PTZ visible/thermal camera | Tower | Records video of the facility *and of public roads* [Legal review] |
| C6 | Edge compute box | Site server room | Fusion, vision model, local alerting, 72 h video buffer; must keep working with the backhaul down |
| C7 | Site network and backhaul | Site → cloud | Fibre or LTE/5G; LTE is jammable and spoofable |
| C8 | Central cloud | Vendor tenant (GovCloud-class) | Multi-site dashboard, evidence store, analytics, model registry |
| C9 | Guard mobile app | Phones or radios carried in the facility | Alert push, map, video clip |
| C10 | Police dispatch handoff | CAD integration / phone | Pilot location and evidence package |
| C11 | ML supply chain | Vendor | Public datasets, synthetic data, training, model signing, OTA |
| C12 | Vendor support access | Vendor → C6/C8 | Remote maintenance |

**Data flows:** sensors → edge box (raw detections, video) → fusion → local alert to guard app (C9) and site console → cloud (alerts, clips, health) → dashboard and dispatch (C10). Models and configuration flow the other way: cloud → edge box.

**Trust boundaries:** (B1) the air interface into C2/C3/C5, which is fully attacker-controlled; (B2) tower equipment to edge box, physically exposed on the perimeter; (B3) site to cloud over the backhaul; (B4) cloud to mobile and CAD; (B5) public internet datasets into the vendor's training pipeline; (B6) vendor staff into production.

## 2. What we protect, in priority order

1. **Availability and integrity of alerting.** A blinded or fooled system that *looks* healthy is the worst outcome, and it is the attacker's goal.
2. **Integrity of evidence:** video, tracks and pilot location, admissible in prosecution with chain of custody.
3. **Confidentiality of coverage:** sensor layout, blind spots, outage windows, alert thresholds. To a smuggling crew, this is worth more than any single alert.
4. **Correctness of pilot location sent to police.** A wrong location means an innocent person gets a police visit.
5. **Bystander privacy:** RF metadata from nearby devices, and video of roads and homes.
6. **Personnel data:** guard identities and shift patterns.
7. **The model and the training data** (intellectual property, and poisoning targets).

## 3. Adversaries

| ID | Who | Capability | Goal |
|---|---|---|---|
| A1 | Smuggling crew / organised crime | Money, consumer and modified drones, cheap SDRs, can buy insider help, learns from failed drops | Deliver contraband undetected; learn and avoid coverage |
| A2 | Corrupt insider (guard or contractor on the payroll) | Physical access, console credentials, knows the schedules | Disable or suppress alerts at drop time; leak blind spots; delete evidence |
| A3 | Inmates with contraband phones | Coordinate drops in real time; may reach the site's Wi-Fi or Bluetooth | Report guard positions and alert status to A1; probe local radios |
| A4 | Remote cybercriminal | Commodity exploits, phishing, ransomware | Ransom the cloud, steal data, use edge boxes as a beachhead |
| A5 | Nation-state (critical-infrastructure customers later) | Advanced persistent threat, supply-chain implants, RF expertise | Pre-position, map defences, blind sensors at a chosen moment |
| A6 | Dataset poisoner | Uploads to public datasets or video sites | Plant a backdoor trigger in the vision model (see [Carlini et al., *Poisoning Web-Scale Training Datasets is Practical*](https://arxiv.org/abs/2302.10149v2)) |
| A7 | Careless hobbyist | Flies near the site without intent | Generates real alerts; must be handled lawfully and proportionately |

## 4. STRIDE by component (top threats only)

**C2/C3 RF and Remote ID (boundary B1)**
- **Spoofing:** forged Remote ID (fake drone, fake operator location) used to create decoys, cause alert fatigue, or redirect police. Receivers have shipped with over-the-air injection bugs ([CVE-2023-31191, CVSS 9.3](https://nozominetworks.com/blog/nozomi-networks-discovers-three-vulnerabilities-affecting-bluemark-dronescout-ds230-remote-id-receiver)). **Control:** Remote ID is never trusted alone. Corroborate with radar and with RF direction-finding to the transmitter, flag inconsistencies as a spoofing indicator, and label any uncorroborated operator location "self-reported, unverified" in the police handoff (see SECURITY_AUDIT P-2).
- **Tampering / denial of service:** malformed frames crashing the parser (fuzz the parser; run it as an unprivileged process in a sandbox with a watchdog) and wideband noise raising the noise floor (detect and alarm on noise-floor anomalies).
- **Evasion:** RF-silent autonomous drones defeat C2/C3 entirely. Accept this, and design radar, acoustic and camera to carry those cases (the simulator shows the dark-drone gap).
- **Information disclosure** [Legal review]: direction-finding captures identifiers (MACs, serials) from devices that are not drones. Minimise at the sensor: keep only drone-protocol frames, hash or truncate others, and enforce retention limits (§7).

**C1 radar, C4 acoustic, C5 camera**
- Blinding or dazzling: lasers or IR floodlights at the camera, decoy heat sources, and radar corner-reflector decoys. **Controls:** an image-health monitor (saturation, sudden loss of contrast, lens-obstruction detection) that raises a *sensor degraded* alarm, never silence; cross-sensor consistency checks (radar track with no camera confirmation becomes a health check, not a dismissal); and filters or shutters on the camera.
- **Physical tampering:** cables cut, sensors turned or repositioned. **Controls:** tamper switches, accelerometers and orientation sensors on mounts, cameras that watch each other's mounts, a heartbeat that raises an alarm within 10 s when lost, and power over UPS.

**C6 edge box**
- If the box is compromised, everything is lost: alerts suppressed, evidence altered, and a pivot point into the facility network. **Controls:**
  - Secure and measured boot with a TPM 2.0, an encrypted disk with the key sealed to that measurement, and a read-only root filesystem.
  - No inbound ports; only outbound mTLS connections to the cloud.
  - A unique device identity per box (keys in the TPM), with signed OTA updates verified on the device (TUF/Uptane-style, with rollback protection).
  - USB ports disabled, a locked enclosure with a tamper switch.
  - Local admin only through break-glass with two-person approval.
- **Fail-safe behaviour:** if the models, any sensor or the fusion process crash or produce garbage, the box alerts "DETECTION DEGRADED" locally *and* to the cloud. The simulator's NaN classifier bug (SECURITY_AUDIT F-03) is exactly the fail-open pattern this rule prevents.

**C7 backhaul**
- Jamming or cutting the backhaul exactly at drop time. **Controls:**
  - The edge box alerts the guard app and site console **locally**, without the cloud.
  - A dual path (fibre plus LTE, or a local radio).
  - The cloud raises a site-offline alarm within 30 s.
  - Store-and-forward of all events with signed timestamps.
- Man-in-the-middle and replay: mTLS everywhere with certificate pinning to a private CA, plus message sequence numbers.

**C8 cloud**
- Account takeover and insider abuse. **Controls:**
  - MFA for every account (phishing-resistant FIDO2 for administrators).
  - Single sign-on and role-based access with least privilege: a site operator sees one site; *only* named security roles can change thresholds or disable sensors, and only with two-person approval.
  - Every configuration change raises an alarm to the department's security office.
- Coverage-map leakage: classify sensor layouts, blind-spot analyses and maintenance windows as **restricted**. Need-to-know access, watermarked exports, no bulk download, and access logged and reviewed monthly.
- Ransomware: immutable, versioned evidence storage (object lock), offline backups, and tested restores.

**C9 guard app and C10 dispatch**
- Stolen phone or lost radio: managed devices (MDM), short sessions, remote wipe, and no evidence cached at rest.
- Alert spoofing to the app: signed push payloads; the app fetches details over mTLS rather than trusting push content.
- Dispatch of a wrong pilot location: every handoff carries method and confidence (`RF direction-finding ±170 m`, `Remote ID self-reported`, `visual`), so officers know what they are walking into.
- Stored XSS: any string that originated over the air (Remote ID serials and IDs) is rendered as text, never as HTML, under a strict CSP (SECURITY_AUDIT P-5).

**C11 ML supply chain (boundary B5)**
- Poisoning or backdoors in public datasets (A6). **Controls:**
  - A manifest with source, license, retrieval date and SHA-256 for every archive, with snapshots stored immutably. Never train on live URLs; the split-view attack in Carlini et al. relies on URLs whose content changes.
  - Deduplication and outlier and label-noise screening.
  - A held-out, vendor-collected, **real** canary test set that public data never touches; block releases on regression.
  - Backdoor-trigger scans (spectral-signature or activation-clustering checks) before each release.
- Malicious model files: pickle-based `.pt` weights run code when loaded. Use ONNX on the edge, hash-pinned dependencies, and signed model artifacts verified at load. The Ultralytics PyPI compromise in Dec 2024 ([Wiz](https://www.wiz.io/blog/ultralytics-ai-library-hacked-via-github-for-cryptomining)) shows this risk is real for this exact stack.
- Adversarial examples and camouflage: an attacker paints or patterns a drone to read as a bird. **Controls:**
  - Fusion must not let the camera *veto* strong radar plus RF plus hover evidence; cap the camera's negative contribution.
  - Adversarial and corruption robustness testing is part of release (red-team task).
  - Thermal plus visible ensembles; behaviour (hover, straight ingress) counts as independent evidence.
- **Alert fatigue as an attack:** A1 sends decoys, toy drones or spoofed Remote ID nightly until guards stop responding, then does the real drop. **Controls:** track false-alarm rates per site and per hour with anomaly alarms; tiered alerts; a camera-confirmed tier that always requires a response; and review of "dismissed" alerts by a second person.

**C12 vendor access**
- Just-in-time, time-boxed access approved by the customer; session recording; no shared accounts; vendor laptops managed and hardened. Supply-chain attestations: an SBOM (CycloneDX/SPDX) per release, SLSA provenance for builds, and signed containers.

**Cross-cutting: evidence chain of custody**
- Hash each video clip and track log at capture on the edge box and sign it with the box's TPM key. Append to a tamper-evident log (hash chain or Merkle log, with periodic anchoring in the cloud). Store originals write-once; export packages with a verification manifest; and log every view and export. Synchronise clocks with authenticated NTP or PTP, plus GPS with spoofing detection: a time skew is an alarm, because a spoofed GNSS time breaks evidence and correlation across sensors.

**Cross-cutting: logging and monitoring**
- Ship security events (logins, configuration changes, sensor health, tamper, OTA, model loads, mTLS failures) to a SIEM, either the customer's or a vendor-operated one with customer read access. Retain for at least a year (CJIS-aligned). Correlation rules: sensor degradation at the same time as a drone alert, a configuration change outside business hours, an operator dismissing a camera-confirmed alert.

## 5. Legal and regulatory constraints that shape the design [Legal review]

| Topic | What we found | Design consequence |
|---|---|---|
| **New SLTT counter-UAS authority** | The SAFER SKIES Act (FY2026 NDAA) and the DHS/DOJ **interim final rule effective July 1, 2026**, codified at 6 U.S.C. 124n, let **state and local law enforcement and correctional agencies** detect, track and identify drones and, with *mitigation certification*, disrupt or disable them. Only agency personnel may operate mitigation; contractors may sell, install, maintain and train. Detection-and-warning certification is online. RF systems must be on a joint **Authorized Systems List**; cameras and acoustic sensors are exempt from the list. Each fixed site needs an Operations Plan (365-day window). The authority sunsets Dec 31, 2031. ([Federal Register, 91 FR 41466](https://www.federalregister.gov/documents/2026/07/06/2026-13609/counter-uas-authority-for-state-local-tribal-and-territorial-law-enforcement-and-correctional); [CRS](https://www.congress.gov/crs-product/IN12661)) | **This supersedes the earlier assumption that only federal agencies can mitigate.** v1 remains detection-only, but (a) the RF detection component should be submitted for the **Authorized Systems List**, since that is likely a sales gate; (b) the product should support the customer's Operations Plan, certification records and the notices the rule requires; (c) a mitigation module operated by agency staff becomes a real roadmap option. |
| **Privacy rules in that IFR** | Control communications may be intercepted only as needed for protection, with minimisation and periodic review (at least quarterly for standing deployments). Records of communications have a **180-day retention limit** with narrow exceptions. Pattern data must be anonymised before sharing outside the agency. No use solely to monitor First Amendment activity. An audit trail is required. (same source, § 124.14) | Build retention enforcement (automatic deletion at 180 days unless there's a legal hold), a minimisation filter at the RF sensor, an anonymised export, an immutable access audit, and a quarterly-review report the customer can file. |
| **FCC implementation** | July 2, 2026 orders: a 180-day nationwide Special Temporary Authority for qualifying SLTT agencies using Authorized Systems List equipment (DA 26-656); an equipment-authorization waiver so vendors can market listed systems (DA 26-657); and declaratory rulings on Section 333 for authorized mitigation and testing (DA 26-655, DA 26-654). **CTIA has petitioned for reconsideration of all four**, so they may change. ([Davis Wright Tremaine](https://www.dwt.com/blogs/broadband-advisor/2026/07/safer-skies-counter-drone-framework)) | Track these proceedings. RF test jamming in our own lab requires an experimental authorization even under DA 26-654. |
| **Federal surveillance statutes (outside 124n authority)** | The 2020 FAA/DOJ/FCC/DHS advisory: radar, electro-optical/IR and acoustic sensors are *less likely* to raise Wiretap or Pen/Trap issues. RF systems that capture communication content, or dialing/routing/addressing information such as serials, MACs, IMSI or IMEI, may implicate the Wiretap Act and the Pen/Trap statute, and private entities cannot obtain pen/trap orders. Radar needs an FCC license. ([advisory](https://www.justice.gov/media/1084061/dl)) | Pilots run on behalf of non-agency customers (malls, stadiums, private utilities) do **not** get 124n relief: ship RF there only after counsel review, or with RF identifier capture disabled. |
| **CJIS Security Policy** | Current version is **v6.1 (released June 25, 2026)**: 256-bit encryption for CJI in transit and at rest outside secure locations, monthly vulnerability scans, MFA for all accounts, zero-trust direction. Priority-1 controls are sanctionable now; others become auditable after Sept 30, 2027. ([SC World](https://www.scworld.com/brief/fbi-cjis-security-policy-updates-to-version-6-1)) | Whether our data is criminal justice information (CJI) is uncertain [Legal review]: drone alerts alone may not be, but integration with CAD, offender records or shared evidence likely pulls the system into scope. **Design to CJIS v6.1 from day one** (AES-256, MFA, monthly scans, US-person personnel screening for cloud administrators, audit logging). Retrofitting would kill state deals. |
| **State law** | States are actively legislating. Examples: Alabama made drone flights over prisons a felony in 2024 ([DroneXL](https://dronexl.co/2024/06/10/flying-drones-alabama-prisons-felony)); a Georgia bill would ban flights within 400 ft of correctional facilities but raises FAA airspace-preemption questions ([Axios, Feb 2026](https://www.axios.com/local/atlanta/2026/02/24/georgia-drone-ban-prisons-jails-contraband-homicides-faa-conflict)); Louisiana gave prisons interception authority ([Corrections1](https://www.corrections1.com/drones/new-la-law-gives-prisons-authority-to-intercept-drones-amid-contraband-concerns)). | Make evidence packages configurable per state (which statute, which jurisdiction's prosecutor). Keep a per-state legal matrix maintained by counsel. |
| **Video of public areas** [Legal review] | Cameras covering perimeter roads and nearby homes may trigger state privacy, camera-registry or biometric laws. Do not add face recognition or license-plate reading without a separate legal review. | Privacy masking of homes and windows in the camera configuration; no biometric processing; retention-limited road footage. |

## 6. Pen-test and red-team plan

| Activity | Scope | Cadence | Notes |
|---|---|---|---|
| Automated vulnerability scanning | Cloud, edge-box images, containers, dependencies | **Monthly** at minimum (CJIS v6.1), plus every release | `pip-audit`, container scans, IaC scans |
| SAST and secret scanning | All repos | Every pull request | Bandit/Semgrep, gitleaks |
| External pen test | Cloud API, dashboard, mobile app, CAD integration | Annually and before major releases | Independent firm; CREST/OSCP-qualified testers |
| Hardware and firmware pen test | Edge box, sensor nodes, Remote ID/RF receivers, OTA | Before v1 GA, then annually | Including over-the-air fuzzing of the RF and Remote ID parsers in a shielded lab |
| ML red team | Vision model and fusion | Every model release | Corruption and adversarial suites, camouflage tests, poisoning canaries, "camera veto" tests |
| **Live-drone red-team exercises** | One pilot site per quarter, then each site twice a year | Scheduled, with the department's written authorisation | Scenarios: night ingress from each approach, RF-silent waypoint flight, decoy plus real drone, swarm of 3, low-altitude behind buildings, spoofed-RID drill in a controlled RF environment only. Requires Part 107-certified pilots, facility airspace coordination (FAA restrictions over some correctional facilities), and **FCC experimental authorisation for any intentional RF interference testing** (DA 26-654). Score: time to alert, drop-zone error, pilot-location error, guard response time. |
| Insider-threat tabletop | Department security plus vendor | Twice a year | "Guard disables sensor before a drop", "coverage map leaked", "evidence deleted" |
| Incident-response exercise | Vendor plus customer | Annually | Ransomware in the cloud; compromised edge box; mass false alarms |
| Bug bounty / vulnerability disclosure | Public VDP from day one, paid bounty after v1 GA | Continuous | security.txt; a safe-harbour policy that excludes live facilities |

**Incident response:** 24/7 on-call; severity tiers (detection loss at any site is SEV-1); customer notification within the contractual SLA and under state breach laws; signed runbooks for "site blind", "edge box suspected compromised" (isolate it, the site falls back to guard procedures, image the disk for forensics, re-provision from a trusted image) and "model suspected poisoned" (roll back to the last signed model, quarantine the dataset snapshot).

## 7. Prioritised controls

**Must have for v1 (before the first state contract):**
1. **Fail-loud detection health:** heartbeats, sensor-degraded and classifier-degraded alarms, and local alerting that works with the backhaul down. Never fail silent.
2. **Remote ID treated as untrusted:** corroborate before confirming a track; every pilot location sent to police carries its method and confidence.
3. **Edge box hardening:** secure and measured boot with a TPM, encrypted disk, read-only root filesystem, no inbound ports, unique device identity.
4. **Signed OTA for software, models and configuration**, verified on the device, with rollback protection.
5. **mTLS on every link** with a private CA; AES-256 at rest and in transit (CJIS v6.1).
6. **Phishing-resistant MFA, role-based access, two-person approval** for threshold, sensor-disable and retention changes, with every change alarmed and audited.
7. **Evidence integrity:** capture-time hashing, signing with the edge box's TPM key, a tamper-evident log, and write-once storage.
8. **Coverage data treated as restricted:** need-to-know access, no bulk export, access reviewed monthly.
9. **ML supply chain:** dataset manifests with hashes and licenses, a held-out real canary set, ONNX inference, signed models, hash-pinned dependencies, an SBOM per release.
10. **Privacy engineering for the 124n rule:** RF minimisation at the sensor, automatic deletion at 180 days with legal-hold support, anonymised exports, and an access audit trail. Counsel review of RF collection before any non-agency deployment.
11. Tamper detection on sensor mounts and enclosures; UPS.
12. SIEM integration and a monthly vulnerability scanning program.
13. A vulnerability disclosure program and a signed incident-response runbook.

**Next (v1.x–v2):**
- Authenticated time (PTP/NTS) with GNSS spoofing detection.
- Hardware security modules for the cloud signing keys.
- Remote ID cross-validation using RF-signature fingerprinting.
- Adversarial training and certified-robustness research.
- Automated alert-fatigue anomaly detection.
- SLSA level 3 builds.
- Confidential-computing VMs for evidence processing.
- FedRAMP or StateRAMP authorisation for the cloud.
- Formal Authorized Systems List submission for the RF module.
- A mitigation module designed for agency-certified operators only.

## 8. Open questions and uncertainties
- Whether alert data and video count as CJI for each customer state. This drives CJIS scope and is a [Legal review] question.
- How far 124n relief extends to *vendor-hosted* processing of RF identifiers on the agency's behalf [Legal review].
- The outcome of CTIA's reconsideration petitions on the FCC orders.
- Federal versus state preemption for state no-fly laws around facilities.
- Whether corrections customers will accept cloud hosting at all, or require on-prem or state-hosted deployments (this changes C8 controls).
