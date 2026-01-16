# Codebase Cleanup Analysis

## Core Files (KEEP)

### Main Application
- `bci_main.py` - Main entry point ✅
- `start_bci.py` - Helper script ✅
- `requirements.txt` - Dependencies ✅

### Source Code (ALL NEEDED)
- `src/` - Entire directory ✅
  - `src/bci/` - BCI modules (interface, classifier, preprocessing, etc.)
  - `src/core/` - Core UPIC functionality
  - `src/gui/` - GUI components
  - `src/utils/` - Utilities

### Active Analysis Tools (KEEP)
- `analyze_motor_imagery.py` - Motor imagery analysis ✅
- `check_data_quality.py` - Data quality checker ✅
- `review_results.py` - Results reviewer ✅

### Configuration & Calibration (KEEP)
- `screen_calibration/` - Screen calibration system ✅
- `references/` - Reference PDFs ✅

### Session Data (KEEP)
- `motor_imagery_sessions/` - Motor imagery session data ✅
- `p300_sessions/` - P300 session data ✅

### Essential Documentation (KEEP)
- `MOTOR_IMAGERY_HANDOFF_GUIDE.md` - Current guide ✅
- `UNICORN_SETUP.md` - Setup instructions ✅

---

## Legacy/Unnecessary Files (DELETE)

### 1. Excess Folder (ENTIRE FOLDER)
- `excess/` - Marked as "excess" ❌
  - `excess/check_unicorn.py`
  - `excess/live_music_test.py`
  - `excess/live_test.py`
  - `excess/main.py`
  - `excess/test_pipeline.py`

### 2. Old Melody Implementation (ENTIRE FOLDER - 167 files!)
- `melody/` - Old P300 melody maker implementation ❌
  - Contains 74 log files, 42 npy files, 35 markdown files
  - Superseded by current `src/bci/` implementation
  - Only keep `melody/README.md` if it has useful info, otherwise delete all

### 3. Old/Backup Source Files
- `src/bci/interface_old.py` - Old version ❌
- `src/bci/preprocessing_old.py` - Old version ❌

### 4. Duplicate/Unclear Files
- `message.py` - Unclear purpose, likely duplicate ❌
- `melody/message.py` - Duplicate ❌

### 5. Separate Online Implementation (REFERENCE ONLY - DELETE)
- `online/` - Separate online MI implementation ❌
  - 8 Python files, 1 JSON
  - This appears to be a reference implementation, not used by main app

### 6. Old Test/Validation Files
- `test_p300_oddball.py` - Old test ❌
- `test_validation_offline.py` - Old validation ❌
- `check_cca_references.py` - Old checker ❌
- `check_performance.py` - Old checker ❌
- `analyze_p300_timings.py` - Old analysis ❌
- `quick_verify_setup.py` - Old verification ❌

### 7. Old Data/Output Files
- `data_/` - Unclear purpose ❌
  - `data_/data.json`
  - `data_/data.wav`
- `demo_output.wav` - Old demo output ❌
- `calibration_data.json` - Old calibration data ❌
- `performance_report.txt` - Old report ❌
- `validation_plots/` - Old validation plots ❌

### 8. Unicorn System Test (REFERENCE - KEEP OR DELETE?)
- `unicorn_system_test/` - Test system for Unicorn
  - Could be useful reference, but might be legacy
  - **DECISION NEEDED**: Keep as reference or delete?

### 9. Outdated Documentation (DELETE)
- `P300_MIGRATION_STATUS.md` - Migration complete ❌
- `P300_PROGRESS_LOG.md` - Old progress log ❌
- `P300_TIMING_ANALYSIS.md` - Old analysis ❌
- `PROGRESS_REPORT.md` - Old report ❌
- `PRE_LAUNCH_CHECKLIST.md` - Launch already done ❌
- `INTEGRATION_VERIFICATION.md` - Verification complete ❌
- `LAUNCH_READY.md` - Launch done ❌
- `VALIDATION_REVIEW.md` - Old review ❌
- `DATA_REVIEW_GUIDE.md` - Old guide ❌
- `CCA_DOWN_ISSUE_DIAGNOSIS.md` - Issue fixed ❌
- `MOTOR_IMAGERY_HANDOFF_GUIDE.md` - **KEEP** (current guide) ✅

### 10. Docs Folder (REVIEW)
- `docs/` - Contains 17 files
  - Some might be outdated checklists
  - **REVIEW NEEDED**: Keep current docs, delete outdated checklists

---

## Summary

### Files to DELETE (High Confidence)
- `excess/` (entire folder)
- `melody/` (entire folder - 167 files)
- `online/` (entire folder)
- `src/bci/interface_old.py`
- `src/bci/preprocessing_old.py`
- `message.py`
- `test_p300_oddball.py`
- `test_validation_offline.py`
- `check_cca_references.py`
- `check_performance.py`
- `analyze_p300_timings.py`
- `quick_verify_setup.py`
- `data_/` (entire folder)
- `demo_output.wav`
- `calibration_data.json`
- `performance_report.txt`
- `validation_plots/` (entire folder)
- All outdated .md files listed above

### Files to REVIEW
- `unicorn_system_test/` - Keep as reference or delete?
- `docs/` - Review which docs are current vs outdated

### Estimated Space Saved
- `melody/` alone: ~167 files
- Total files to delete: ~200+ files
