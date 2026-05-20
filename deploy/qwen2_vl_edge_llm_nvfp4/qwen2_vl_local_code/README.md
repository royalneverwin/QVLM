# Qwen2-VL Local Code

This directory is populated by `x86_host/00_prepare_local_model.sh`.

It serves two purposes:

- keep an editable copy of Qwen2-VL code inside the QVLM repo
- provide code files that are symlinked into the prepared local model folder

The prepare script seeds these files from the installed `transformers`
package, rewrites imports for local dynamic loading, and then links them into
the downloaded model directory.

Edit the files in this directory when you later add `VisionZip` and
`QAPruner`.
