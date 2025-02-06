# 🚨 **Threat Detection and Mitigation System with SDN Integration** 🚨

## 📖 **Overview**
This project implements a comprehensive **Threat Detection and Mitigation System** using machine learning models to detect threats and **Software-Defined Networking (SDN)** to dynamically mitigate them. The system integrates:

- **Threat Detection**: An ensemble model for precise threat identification.
- **SDN Integration**: Real-time mitigation strategies like rate limiting, network segmentation, and queuing-based traffic management.

---

## 🛠️ **Features**

- **🔍 Data Preprocessing**: 
  - Encodes data
  - Applies wavelet transforms for signal processing
  - Standardizes features for optimal model performance

- **🚨 Detection**:
  - Utilizes an ensemble of machine learning models for high precision threat detection

- **🌐 SDN-Integrated Mitigation**:
  - Open vSwitch (OVS) for dynamic network-level adjustments like queuing and rate limiting

- **⚡ Advanced Mitigation Techniques**:
  - Rate limiting
  - Firewalls
  - Intrusion Prevention Systems (IPS)
  - Network segmentation
  - Automated responses

- **📊 Visualization**:
  - Graphs for model performance and traffic flow analysis

---

## 🏁 **Getting Started**

### 🖥️ **Prerequisites**:
- Python 3.7+
- WSL (for Linux-based environments) or Linux server
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pywavelets`
- **Open vSwitch (OVS)** installed and configured for SDN-based mitigation

---

## 🔍 **Code Walkthrough**

### **Detection**:
The detection module applies a weighted ensemble of 5 machine learning models:
- **Logistic Regression**
- **Random Forest**
- **SVM**
- **Gradient Boosting**
- **KNN**

Each model is trained on preprocessed data, and the predictions are combined using optimized weights for high accuracy.

---

### **Mitigation**:
The mitigation phase includes:
- **🔒 Rate Limiting**: Controls suspicious traffic by limiting bandwidth
- **🛡️ Intrusion Prevention System (IPS)**: Blocks potential threats
- **🌐 Web Application Firewall (WAF)**: Protects against malicious HTTP requests
- **🔌 Network Segmentation**: Isolates critical resources for added security
- **⚙️ Automated Response**: Executes predefined actions against threats

---

### **SDN Integration**:
The SDN component provides dynamic responses through **Open vSwitch (OVS)**:
- **🔒 Rate Limiting via SDN**: Configures OVS rules to restrict suspicious traffic at the switch level
- **🔄 Traffic Queuing**: Uses queuing theory to balance load and prevent congestion via OVS QoS rules

---

## 📈 **Results**

- **Detection Accuracy**: The ensemble model achieves high detection accuracy and **F1 score**.
- **SDN Response**: The SDN-based response dynamically adjusts network settings, ensuring a rapid response.
- **Visualization**: Performance graphs demonstrate the accuracy and effectiveness of detection and mitigation strategies.

---

## 🔮 **Future Scope**

- **⚡ Real-Time SDN Monitoring**: Integrate live data feeds for dynamic SDN policy adjustments
- **🧠 Adaptive SDN Rules**: Use machine learning for real-time adjustments based on evolving threats
- **🚀 Enhanced Mitigation**: Expand SDN capabilities to deploy distributed queuing and AI-powered responses

---

## 🛠️ **Troubleshooting**

- **OVS Commands**: Verify OVS installation and interface configuration if SDN functions fail
- **Data Processing**: Ensure the integrity and compatibility of datasets for preprocessing

---

Feel free to explore and contribute to this project! 😊
