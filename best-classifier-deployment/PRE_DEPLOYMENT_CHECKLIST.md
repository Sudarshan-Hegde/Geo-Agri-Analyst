# Pre-Deployment Checklist for BestClassifier

Complete this checklist BEFORE attempting deployment to avoid common issues.

## ✅ Phase 1: Model Verification

- [ ] **Model file exists**
  ```bash
  ls -lh bestClassifier.pth
  # Expected: 100-500 MB
  ```

- [ ] **Model loads successfully**
  ```python
  import torch
  state_dict = torch.load('bestClassifier.pth', map_location='cpu')
  print(f"Keys: {len(state_dict)}")
  print(f"First 5 keys: {list(state_dict.keys())[:5]}")
  ```

- [ ] **State dict keys documented**
  - Save first 20 keys to a file for reference
  - Check for 'module.' prefix (indicates DataParallel)
  - Note layer naming pattern (trunk_a vs rrdb_trunk)

## ✅ Phase 2: Architecture Documentation

- [ ] **Training notebook identified**
  - Location: `_______________________`
  - Last modified: `_______________________`

- [ ] **Model classes extracted**
  - [ ] RFB class copied
  - [ ] RRDB class copied  
  - [ ] RRFDB class copied (if used)
  - [ ] Generator class copied
  - [ ] Classifier class copied

- [ ] **Architecture parameters documented**
  ```python
  # Example - fill in your values:
  num_rrdb = ___  # Usually 12
  num_rrfdb = ___  # Usually 6
  nf = ___  # Usually 64
  num_classes = ___  # Usually 19
  ```

- [ ] **Layer names match**
  - Compare app.py layer names with state_dict keys
  - Fix any mismatches BEFORE deployment

## ✅ Phase 3: Preprocessing Verification

- [ ] **Input size documented**
  - Training input size: `_______`
  - Expected format: RGB / Grayscale
  - Value range: 0-1 / 0-255 / -1 to 1

- [ ] **Normalization parameters**
  ```python
  # Copy from training:
  mean = [___, ___, ___]  # or None
  std = [___, ___, ___]   # or None
  ```

- [ ] **Transform pipeline copied**
  - Resize operations
  - Normalization
  - ToTensor
  - Any custom transforms

## ✅ Phase 4: Class Labels

- [ ] **Class names extracted**
  ```python
  CLASS_NAMES = [
      # List all 19 classes in correct order
  ]
  ```

- [ ] **Label order verified**
  - Matches training notebook
  - Matches validation results
  - Saved to `label_indices.json`

- [ ] **label_indices.json created**
  ```json
  {
    "class_names": ["class0", "class1", ...],
    "num_classes": 19
  }
  ```

## ✅ Phase 5: Dependencies

- [ ] **PyTorch version identified**
  - Training used: `torch==_____`
  - Match in requirements.txt

- [ ] **Torchvision version identified**
  - Training used: `torchvision==_____`
  - Match in requirements.txt

- [ ] **All imports documented**
  - List any custom dependencies
  - Add to requirements.txt

## ✅ Phase 6: Local Testing

- [ ] **app.py customized**
  - Model architecture copied
  - Preprocessing implemented
  - Class names updated
  - All TODOs resolved

- [ ] **Local test successful**
  ```bash
  cd best-classifier-deployment
  python app.py
  # Visit http://localhost:7860
  # Upload test image
  # Verify predictions
  ```

- [ ] **Test predictions make sense**
  - Not all same class
  - Confidence scores reasonable (not all 0 or 1)
  - Top prediction matches visual inspection

## ✅ Phase 7: Deployment Files Ready

- [ ] **app.py complete** (no TODOs remaining)
- [ ] **requirements.txt finalized**
- [ ] **README.md customized**
- [ ] **label_indices.json present**
- [ ] **bestClassifier.pth ready**

## ✅ Phase 8: HuggingFace Setup

- [ ] **HuggingFace account created**
  - Username: `_______________________`
  - Email verified: Yes / No

- [ ] **Space created**
  - Name: `bestClassifier`
  - SDK: Gradio
  - License: MIT
  - Visibility: Public / Private

- [ ] **Git LFS installed**
  ```bash
  git lfs install
  git lfs version
  ```

- [ ] **HuggingFace CLI installed**
  ```bash
  huggingface-cli --version
  ```

- [ ] **Authenticated**
  ```bash
  huggingface-cli login
  huggingface-cli whoami
  ```

## ✅ Phase 9: Pre-Flight Checks

- [ ] **Disk space available** (at least 2GB)
  ```bash
  df -h .
  ```

- [ ] **Internet connection stable**

- [ ] **Deployment script executable**
  ```bash
  chmod +x deploy_bestclassifier.sh
  ```

- [ ] **Backup created**
  ```bash
  cp bestClassifier.pth bestClassifier.pth.backup
  ```

## ✅ Phase 10: Risk Assessment

Review potential issues:

- [ ] **Model size acceptable** (<500MB recommended)
  - If >500MB: Consider quantization
  - If >1GB: Must split or compress

- [ ] **Architecture complexity**
  - Simple models: Lower risk
  - Complex custom layers: Higher risk
  - Custom operations: May need CPU-only

- [ ] **Dependencies conflicts**
  - Check for version incompatibilities
  - Test in clean environment if possible

- [ ] **Expected inference time**
  - CPU: <5 seconds acceptable
  - If >10 seconds: Consider optimization

## ⚠️ Known Issues to Avoid

Based on previous deployments:

- [ ] **Gradio version** - Use >=4.44.1 (not 4.44.0)
- [ ] **Layer name mismatch** - Double-check state_dict keys
- [ ] **DataParallel prefix** - Handle 'module.' stripping
- [ ] **Server config** - Include server_name="0.0.0.0"
- [ ] **Eval mode** - Ensure model.eval() is called
- [ ] **Device handling** - Use map_location in torch.load()

## 🎯 Ready to Deploy?

If ALL checkboxes are ticked:

```bash
./deploy_bestclassifier.sh
```

If ANY checkbox is unchecked:
- Complete that step first
- Do NOT proceed with deployment
- Refer to BESTCLASSIFIER_DEPLOYMENT_GUIDE.md

---

**Checklist completed by:** `_______________________`  
**Date:** `_______________________`  
**Estimated deployment time:** 5-10 minutes (build time)  
**Expected outcome:** Working Gradio interface with predictions
