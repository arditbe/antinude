# 🧠 Conservative NudeNet API (Flask)

A **lightweight, conservative NSFW detection API** built with [Flask](https://flask.palletsprojects.com/) and [NudeNet](https://github.com/notAI-tech/NudeNet).  
This server only flags **explicit nudity or sexual acts**, not mere skin exposure.

---

## 🚀 Overview

This Flask API uses `nudenet.NudeDetector` to analyze uploaded images and determine whether they contain **explicit sexual or nude content** according to strict, conservative rules.

It intentionally ignores:
- Bare male torsos  
- Swimsuits, bikinis, underwear  
- Non-explicit skin exposure  

It only marks **"nude"** if explicit exposure or sexual acts are detected.

---

## ⚙️ Features

✅ CPU-only inference (no CUDA required)  
✅ Conservative classification rules  
✅ Automatic temp file cleanup  
✅ JSON-based REST API  
✅ Debug mode returning raw labels and method used  

---

## 🧩 Requirements

- Python 3.8+
- [Flask](https://pypi.org/project/Flask/)
- [Pillow](https://pypi.org/project/Pillow/)
- [NudeNet](https://pypi.org/project/nudenet/)

Install dependencies:

```bash
pip install flask pillow nudenet
