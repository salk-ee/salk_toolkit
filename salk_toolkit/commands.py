"""CLI Commands
-------------

This module replaces `11_commands.ipynb` and defines the small number of
command-line entry points we ship with the package (Explorer launcher,
translation helpers, etc.).
"""

__all__ = [
    "streamlit_fn_factory",
    "translate_pot",
    "translate_dashboard_fn",
    "translate_dashboard",
]

# Keep this list minimal as this py will actually be executed
import os
import sys


# --------------------------------------------------------
#          STK EXPLORER
# --------------------------------------------------------
# Run explorer Streamlit app from anywhere with the `stk_explorer` command.
def streamlit_fn_factory(relpath, curpath):
    def run_streamlit_fn_fn():
        import subprocess

        filename = os.path.join(curpath, relpath)

        subprocess.run(["streamlit", "run", filename] + sys.argv[1:])

    return run_streamlit_fn_fn


# | eval: false
# Run explorer app
run_explorer = streamlit_fn_factory("./tools/explorer.py", os.path.dirname(__file__))


# Translate a pot file using generic tfunc
# Could be useful if you don't want to use deepl
def translate_pot(template, dest, t_func, sources=[]):
    import polib
    from tqdm import tqdm
    from collections import defaultdict

    pot = polib.pofile(template)

    if os.path.exists(dest):
        po = polib.pofile(dest)
        if dest not in sources:
            sources.append(dest)  # For copying between contexts
    else:
        po = polib.POFile()
        po.metadata = pot.metadata

    todo = defaultdict(list)

    existing = {(entry.msgctxt, entry.msgid) for entry in po}

    for entry in pot:
        if (entry.msgctxt, entry.msgid) in existing:
            continue
        todo[entry.msgid].append(entry)

    # Go through sources and add translations found there to the pot
    if sources and len(todo) > 0:
        n_existing = len(existing)
        for source in tqdm(sources, desc="Checking existing translations"):
            spo = polib.pofile(source)
            for entry in spo:
                if entry.msgid not in todo:
                    continue

                for tentry in todo[entry.msgid]:
                    tentry.msgstr = entry.msgstr
                    tentry.tcomment = entry.tcomment
                    po.append(tentry)
                    existing.add((tentry.msgctxt, tentry.msgid))
                del todo[entry.msgid]

        n_found = len(existing) - n_existing
        if n_found:
            print(f"Found {n_found} translations in {sources}")

    progress = tqdm(pot, desc="Translating", total=len(todo))

    try:
        for msgid in todo:
            msgstr = todo[msgid][0].msgstr
            if not msgstr:
                continue
            tmsgstr = t_func(msgstr)
            for tentry in todo[msgid]:
                tentry.msgstr = tmsgstr
                po.append(tentry)
            progress.update(1)

    except KeyboardInterrupt:  # Ctrl-c
        print("Keyboard interrupt, finishing early and saving partial results")

    progress.close()

    po.save(dest)


# --------------------------------------------------------
#          TRANSLATE DASHBOARD
# --------------------------------------------------------
# Use Deepl to translate a dashboard with the `translate_stk_dashboard` command.
def translate_dashboard_fn(dashboard_file, target_lang, deepl_key, context=None, source_lang="en"):
    import deepl  # requires this, but not installed with salk_toolkit

    apppath = os.path.splitext(dashboard_file)[0]
    path, app = os.path.split(apppath)

    translator = deepl.Translator(deepl_key)

    def t_func(txt):
        return translator.translate_text(txt, source_lang=source_lang, target_lang=target_lang, context=context).text

    print(f"Translating {app} to {target_lang}")

    locale_dir = os.path.join(path, f"locale/{target_lang}")
    if not os.path.exists(locale_dir):
        os.mkdir(locale_dir)

    pot_loc = os.path.join(path, f"locale/{app}.pot")
    po_loc = os.path.join(path, f"locale/{target_lang}/{app}.po")

    print(f"Template file: {pot_loc}")
    print(f"Result file: {po_loc}")

    # Find other pot files in locale/{target_lang}/ to use as translation sources
    sources = []
    for f in os.listdir(locale_dir):
        if f.endswith(".po") and f != f"{app}.po":
            sources.append(os.path.join(locale_dir, f))
    if len(sources) > 0:
        slist = [os.path.basename(s) for s in sources]
        print(f"Using {len(sources)} extra sources: {', '.join(slist)}")

    translate_pot(pot_loc, po_loc, t_func, sources)


def translate_dashboard():
    if len(sys.argv) < 4:
        print("Requires three parameters: <deepl auth key> <dashboard file name> <language>")
        print("Additional parameters are <'context'> <source language>")
        sys.exit()

    deepl_key = sys.argv[1]
    dashboard_file = sys.argv[2]
    target_lang = sys.argv[3]
    context = sys.argv[4] if len(sys.argv) > 4 else None
    source_lang = sys.argv[5] if len(sys.argv) > 5 else "en"

    translate_dashboard_fn(dashboard_file, target_lang, deepl_key, context=context, source_lang=source_lang)
