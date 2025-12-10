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
    "run_annotator",
]

# Keep this list minimal as this py will actually be executed
import os
import sys
from typing import Callable


# --------------------------------------------------------
#          STK EXPLORER
# --------------------------------------------------------
# Run explorer Streamlit app from anywhere with the `stk_explorer` command.
def streamlit_fn_factory(relpath: str, curpath: str) -> Callable[[], None]:
    """Create a function that runs a Streamlit app at the given path.

    Args:
        relpath: Relative path to the Streamlit app file.
        curpath: Current directory path where the app should be run from.

    Returns:
        A callable that executes the Streamlit app when called.
    """

    def _run_streamlit_fn_fn() -> None:
        import subprocess

        filename = os.path.join(curpath, relpath)

        subprocess.run(["streamlit", "run", filename] + sys.argv[1:])

    return _run_streamlit_fn_fn


# | eval: false
# Run explorer app
run_explorer = streamlit_fn_factory("./tools/explorer.py", os.path.dirname(__file__))
# Run annotator app
run_annotator = streamlit_fn_factory("./tools/annotator.py", os.path.dirname(__file__))


def translate_pot(template: str, dest: str, t_func: Callable[[str], str], sources: list[str] | None = None) -> None:
    """Translate a pot file using generic tfunc.

    Could be useful if you don't want to use deepl.
    Translate a .pot file using a custom translation function.

    Args:
        template: Path to the source .pot template file.
        dest: Path where the translated .po file should be written.
        t_func: Translation function that takes a source string and returns translated string.
        sources: Optional list of existing .po files to use as translation sources.
    """
    if sources is None:
        sources = []
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
def translate_dashboard_fn(
    dashboard_file: str, target_lang: str, deepl_key: str, context: str | None = None, source_lang: str = "en"
) -> None:
    """Translate a dashboard using DeepL API.

    Args:
        dashboard_file: Path to the dashboard file to translate.
        target_lang: Target language code (e.g., 'cs', 'lt').
        deepl_key: DeepL API authentication key.
        context: Optional context string for better translations.
        source_lang: Source language code (default: 'en').
    """
    import deepl  # requires this, but not installed with salk_toolkit

    apppath = os.path.splitext(dashboard_file)[0]
    path, app = os.path.split(apppath)

    translator = deepl.Translator(deepl_key)

    def _t_func(txt: str) -> str:
        """Translate text using DeepL API."""
        result = translator.translate_text(txt, source_lang=source_lang, target_lang=target_lang, context=context)
        # translate_text returns TextResult for single input, list[TextResult] for batch
        if isinstance(result, list):
            return result[0].text
        return result.text

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

    translate_pot(pot_loc, po_loc, _t_func, sources)


def translate_dashboard() -> None:
    """CLI entry point for dashboard translation.

    Reads command-line arguments and calls translate_dashboard_fn.
    Requires at least 3 arguments: <deepl_key> <dashboard_file> <target_lang>
    Optional 4th and 5th arguments: <context> <source_lang>
    """
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
