import os
os.environ["ORT_NO_OPENCL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, request, jsonify
import tempfile, os, traceback, logging
from PIL import Image
try:
    from nudenet import NudeDetector
except Exception:
    NudeDetector = None


SEXUAL_ACT_THRESHOLD = 0.55  
LABEL_CONFIDENCE_MIN = 0.40  
MAX_UPLOAD_BYTES = 80 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES

if NudeDetector is None:
    logging.error("nudenet.NudeDetector not importable. Install nudenet in the venv.")
    raise SystemExit("nudenet import failed")

logging.info("Loading NudeNet model...")
try:
    model = NudeDetector()
    logging.info("NudeNet model instance created: %s", type(model))
except Exception:
    logging.exception("Failed to instantiate NudeDetector")
    raise
def try_model_methods(path):
    """Try several method names on the NudeNet instance and return (result, method_name)."""
    candidates = ['predict','classify','detect','classify_image','predict_image','predict_file','classify_files','__call__']
    last_exc = None
    for name in candidates:
        if hasattr(model, name):
            func = getattr(model, name)
            try:
                logging.info("Attempting model.%s on %s", name, path)
                res = func(path)
                logging.info("model.%s succeeded", name)
                return res, name
            except Exception as e:
                logging.exception("model.%s failed", name)
                last_exc = e
    raise RuntimeError(f"No model method succeeded. Last: {last_exc}")

def normalize_score_entry(entry):
    if entry is None:
        return {'safe': 1.0, 'unsafe': 0.0}
    if isinstance(entry, dict):
        if 'safe' in entry or 'unsafe' in entry:
            return {'safe': float(entry.get('safe', 0.0)), 'unsafe': float(entry.get('unsafe', 0.0))}
        for k, v in entry.items():
            if isinstance(v, (float, int)):
                kl = k.lower()
                if 'unsafe' in kl or 'nsfw' in kl or 'porn' in kl:
                    val = float(v)
                    return {'safe': 1.0 - val, 'unsafe': val}
                if 'safe' in kl:
                    val = float(v)
                    return {'safe': val, 'unsafe': 1.0 - val}
    if isinstance(entry, (float, int)):
        val = max(0.0, min(1.0, float(entry)))
        return {'safe': 1.0 - val, 'unsafe': val}
    if isinstance(entry, (list, tuple)):
        if len(entry) == 2 and all(isinstance(x, (int, float)) for x in entry):
            return {'safe': float(entry[0]), 'unsafe': float(entry[1])}
    return None
def extract_labels_from_result(result):
    labels = []

    def scan_item(item):
        if not isinstance(item, dict):
            return
        lbl = (item.get('label') or item.get('class') or item.get('name') or '').strip()
        conf = None
        for k in ('score','confidence','prob','probability'):
            if k in item and isinstance(item[k], (int, float)):
                conf = float(item[k])
                break
        if conf is None and 'scores' in item and isinstance(item['scores'], dict):
            for sk in ('unsafe','nsfw','porn','nudity'):
                if sk in item['scores']:
                    val = item['scores'][sk]
                    if isinstance(val, (int, float)):
                        conf = float(val); break
        if lbl:
            labels.append((lbl, float(conf) if conf is not None else None))
    if isinstance(result, dict):
        for k,v in result.items():
            if isinstance(v, list):
                for it in v:
                    scan_item(it)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, list):
                        for it in vv: scan_item(it)
            else:
                if isinstance(k, str) and '/' not in k and isinstance(v, (float, int)):
                    labels.append((k, float(v)))
    elif isinstance(result, list):
        for item in result:
            scan_item(item)
    else:
        try:
            for item in list(result):
                if isinstance(item, dict):
                    scan_item(item)
        except Exception:
            pass
    label_map = {}
    for lbl, conf in labels:
        nl = lbl.strip().lower()
        if nl == '': continue
        if conf is None:
            conf = 0.0
        prev = label_map.get(nl)
        if prev is None or conf > prev:
            label_map[nl] = conf
    out = [(k, v) for k, v in label_map.items()]
    return out

def label_means_exposed_nude(lbl):
    """
    HOW THE ALGORITHM CHECKS:
     - For WOMEN: exposed breasts/areola/nipple OR exposed genitals -> positive
     - For MEN: male breast/torso exposures are NOT positive; only exposed male genitals (penis) -> positive
     - Sexual act labels (oral, blowjob, sex, intercourse, etc.) -> positive
    This function operates on normalized lowercased label strings.
    """
    if not lbl:
        return False

    l = lbl.lower()
    if any(tok in l for tok in ('covered', 'cover', 'bra', 'underwear', 'bikini', 'swimsuit', 'shirt', 'clothe', 'dressed')):
        return False
    sexual_tokens = ['oral', 'blowjob', 'fellatio', 'sucking', 'cunnilingus', 'sex', 'intercourse', 'penetration', 'masturbation', 'porn', 'cum']
    if any(tok in l for tok in sexual_tokens):
        return True
    if 'penis' in l or 'male_genital' in l or 'male_genitalia' in l:
        return True
    if ('vagina' in l or 'pussy' in l or 'female_genital' in l or 'female_genitalia' in l or 'labia' in l) and ('expos' in l or 'exposed' in l or 'visible' in l or 'open' in l):
        return True
    if any(tok in l for tok in ('female_breast_exposed', 'female_nipple_exposed', 'areola', 'areola_exposed')):
        return True
    if 'female' in l and any(tok in l for tok in ('breast', 'nipple', 'areola')) and any(tok in l for tok in ('expos', 'exposed', 'visible', 'bare')):
        return True
    return False
@app.route("/", methods=['GET'])
def index():
    return jsonify({'status': 'ok', 'message': 'exlipt Exposure'})

@app.route("/check", methods=['POST'])
def check_image():
    temp_path = None
    try:
        upload_field = 'image' if 'image' in request.files else ('file' if 'file' in request.files else None)
        if not upload_field:
            return jsonify({'status': 'error', 'message': 'No file uploaded. Use field "image" or "file".'}), 400

        file = request.files[upload_field]
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty filename'}), 400
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".jpg")
        temp_path = tmp.name
        tmp.close()
        file.save(temp_path)
        logging.info("Saved upload to %s (size=%d)", temp_path, os.path.getsize(temp_path))
        try:
            model_result, method_used = try_model_methods(temp_path)
        except Exception as e:
            logging.exception("Model call failed; returning safe by default")
            return jsonify({'status': 'ok', 'verdict': 'safe', 'scores': {'safe': 1.0, 'unsafe': 0.0}, 'debug': {'method_error': str(e)}}), 200
        labels = extract_labels_from_result(model_result)
        debug_labels = [{'label': l, 'confidence': c} for (l, c) in labels]

        labels_filtered = [(l, c) for (l, c) in labels if c is None or c >= LABEL_CONFIDENCE_MIN]
        positive_labels = []
        for (l, c) in labels_filtered:
            if label_means_exposed_nude(l):
                lc = l.lower()
                if any(tok in lc for tok in ('oral','blowjob','fellatio','sucking','cunnilingus','sex','intercourse','penetration','masturbation','porn')):
                    if c is not None and c < SEXUAL_ACT_THRESHOLD:
                        continue
                positive_labels.append({'label': l, 'confidence': c})

        debug = {
            'method_used': method_used,
            'labels_raw': debug_labels,
            'labels_filtered': [{'label': l, 'confidence': c} for (l, c) in labels_filtered],
            'positive_labels': positive_labels
        }
        if len(positive_labels) > 0:
            return jsonify({'status': 'ok', 'verdict': 'nude', 'scores': {'safe': 0.0, 'unsafe': 1.0}, 'debug': debug}), 200
        return jsonify({'status': 'ok', 'verdict': 'safe', 'scores': {'safe': 1.0, 'unsafe': 0.0}, 'debug': debug}), 200

    except Exception as exc:
        logging.exception("Unhandled exception in /check")
        tb = traceback.format_exc()
        return jsonify({'status': 'error', 'message': 'internal_server_error', 'detail': str(exc), 'trace': tb}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                logging.exception("Failed to remove temp file")

if __name__ == "__main__":
    logging.info("THIS IS FOR DEVELOPING DEBUGING ONLY, USE A WSGI SERVER MR DEVELOPER OR U WILL BE FFCKD 127.0.0.1:6969")
    app.run(host='127.0.0.1', port=6969, debug=False, threaded=True)
