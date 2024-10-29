Threat Detection and Mitigation System with SDN Integration
Overview
This project implements a comprehensive Threat Detection and Mitigation System using machine learning models to detect threats and software-defined networking (SDN) to dynamically mitigate detected threats. The detection component combines an ensemble model for precise threat identification, while the mitigation leverages SDN to apply dynamic response measures like rate limiting, network segmentation, and queuing-based traffic management for efficient response to real-time threats.

Features
Data Preprocessing: Encodes data, applies wavelet transforms for signal processing, and standardizes features.
Detection: Utilizes an ensemble model, combining the strengths of several machine learning classifiers.
SDN-Integrated Mitigation: Incorporates SDN for network-level adjustments like queuing and rate limiting using Open vSwitch (OVS).
Advanced Mitigation Techniques: Includes rate limiting, firewalls, intrusion prevention systems, network segmentation, and automated responses.
Visualization: Offers graphs for model performance and traffic flow to analyze threat detection and mitigation.

Getting Started
Prerequisites
Python 3.7+
WSL (for Linux-based environments) or Linux server
Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, pywavelets
Open vSwitch (OVS) installed and configured for SDN-based mitigation

Code Walkthrough
Detection
The detection part applies a weighted ensemble of five machine learning models—Logistic Regression, Random Forest, SVM, Gradient Boosting, and KNN. Each model is trained on preprocessed data, and predictions are combined based on optimized weights. This approach ensures high accuracy by leveraging the strengths of each algorithm.

Mitigation
The mitigation phase includes various methods:

Rate Limiting: Controls traffic from suspicious sources to limit bandwidth.
Intrusion Prevention System (IPS): Monitors and blocks threats.
Web Application Firewall (WAF): Protects against malicious HTTP requests.
Network Segmentation: Isolates critical network resources.
Automated Response: Executes predefined responses to threats.
SDN Integration
The SDN component allows dynamic response through Open vSwitch, providing:

Rate Limiting via SDN: Configures OVS rules to restrict suspicious traffic at the switch level.
Traffic Queuing: Implements queuing theory to balance load and prevent network congestion using OVS QoS rules.
Results
Detection Accuracy: The ensemble model reaches high detection accuracy and F1 score.
SDN Response: SDN-based response actions dynamically adjust network settings, ensuring rapid response.
Visualization: Performance graphs illustrate the accuracy and efficacy of detection and response.

Future Scope
Real-Time SDN Monitoring: Integrate live data feeds to dynamically adjust SDN policies.
Adaptive SDN Rules: Use machine learning for real-time adjustment of SDN policies based on threat types.
Enhanced Mitigation: Expand SDN functions, such as deploying distributed queuing and AI-powered response.
Troubleshooting
OVS Commands: Check OVS installation and interface configuration if SDN functions fail.
Data Processing: Ensure dataset integrity and compatibility with preprocessing steps.
