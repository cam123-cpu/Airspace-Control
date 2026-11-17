"""Static guarantee: no SQL is ever built from strings. Every call that executes SQL must receive a string
LITERAL as its statement (values go only through bound parameters)."""
import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / "airspace"
SQL_CALLS = {"q", "one", "x", "execute", "executemany", "executescript"}
# Only two non-literals are allowed, both inside db.py: the DB wrapper's q()/one()/x() pass their `sql` argument
# through (their callers are what this scan checks), and _apply() runs the constant MIGRATIONS scripts.
ALLOWED_NAMES = {("db.py", "script"), ("db.py", "sql")}


def _violations():
    out = []
    for f in sorted(PKG.glob("*.py")):
        tree = ast.parse(f.read_text(), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in SQL_CALLS and node.args:
                a = node.args[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    continue
                if isinstance(a, ast.Name) and (f.name, a.id) in ALLOWED_NAMES:
                    continue
                out.append(f"{f.name}:{node.lineno} {node.func.attr}({ast.unparse(a)[:60]})")
    return out


def test_no_dynamic_sql():
    v = _violations()
    assert not v, "SQL must be a string literal:\n" + "\n".join(v)


def test_scanner_catches_bad_patterns(tmp_path):
    bad = ['db.q(f"SELECT * FROM t WHERE id = {x}")', 'db.x("SELECT " + col)', 'cur.execute("SELECT %s" % v)', 'db.one(sql)']
    for src in bad:
        tree = ast.parse(src)
        call = tree.body[0].value
        a = call.args[0]
        assert not (isinstance(a, ast.Constant) and isinstance(a.value, str)), src


def test_no_dangerous_calls():
    banned = {"eval", "exec", "pickle", "marshal", "os.system", "subprocess", "yaml.load", "shell=True"}
    for f in PKG.glob("*.py"):
        src = f.read_text()
        for b in banned:
            assert b + "(" not in src and (b != "shell=True" or b not in src), f"{b} in {f.name}"
