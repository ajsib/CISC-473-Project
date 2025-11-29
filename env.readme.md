# **Environment Setup — README**

## **Requirements**

* Python **3.10–3.11**
* Git
* A machine with either:

  * NVIDIA GPU with CUDA drivers installed, or
  * CPU-only execution (slower)

No other global tools are required. The project manages its environment automatically.

---

## **1. Clone the Repository**

```bash
git clone <repo-url>
cd <repo>
```

---

## **2. Select the Python Version**

Use any method you prefer. Examples:

**pyenv**

```bash
pyenv install 3.10.14
pyenv local 3.10.14
```

**system Python**

```bash
python3 --version  # must show 3.10.x or 3.11.x
```

---

## **3. Launch the CLI**

Run the project’s interactive manager:

```bash
python -m src.cli.main
```

The first time you run it, the environment wizard will appear and build everything for you.

---

## **4. Build the Environment (Automatic)**

From inside the CLI:

```
[7] Environment wizard
```

It will:

1. Inspect the interpreter
2. Create `.venv`
3. Install all required packages
4. Verify GPU / CPU support
5. Report readiness

No manual pip or conda commands are needed.

---

## **5. Running the Pipeline**

Once the environment passes:

**Run everything (S1–S7):**

```bash
python -m src.cli.main --stage=all
```

**Or use the menu:**

```
[1] Run full pipeline
```

**Run a single stage:**

```
[2] Run single stage
```

---

## **6. Workspace**

All outputs, tables, figures, and logs go in:

```
results/
```

To reset:

```
[5] Clear workspace
[6] Clear workspace (full)
```

---

## **7. Logs**

To view recent pipeline activity:

```
[4] Show last 50 log lines
```

---

## **8. Summary**

The only commands a new user needs:

```bash
git clone <repo>
cd <repo>

# ensure Python 3.10 or 3.11
python -m src.cli.main
```

The CLI takes care of everything else—environment, verification, execution, and management.
