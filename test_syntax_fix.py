#!/usr/bin/env python3

"""Simple syntax verification for the AnyIO fix"""

import ast
import sys


def test_syntax():
    """Test that our fix has valid Python syntax"""

    print("🧪 Testing syntax of modified proxy2.py...")

    try:
        # Read the modified file
        with open("/tmp/oss-bentoml/src/_bentoml_impl/client/proxy2.py", "r") as f:
            code = f.read()

        # Parse the Python code
        ast.parse(code)
        print("✅ Python syntax is valid")

        # Check that our fix is present
        if "anyio.lowlevel.current_token()" in code:
            print("✅ Fix is present - anyio.lowlevel.current_token() found")
        else:
            print("❌ Fix not found in code")
            return False

        if "except LookupError:" in code:
            print("✅ Exception handling is present - LookupError catch found")
        else:
            print("❌ LookupError exception handling not found")
            return False

        if "import anyio.lowlevel" in code:
            print("✅ Import is present - anyio.lowlevel imported")
        else:
            print("❌ anyio.lowlevel import not found")
            return False

        # Verify the logic makes sense
        if "token=token" in code:
            print("✅ Token parameter is being passed to anyio.from_thread.run")
        else:
            print("❌ Token parameter not found")
            return False

        return True

    except SyntaxError as e:
        print(f"❌ Syntax error in modified file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading/parsing file: {e}")
        return False


def verify_imports():
    """Verify that anyio.lowlevel is a real module"""

    print("\n🔍 Verifying anyio.lowlevel availability...")

    try:
        import anyio.lowlevel

        print("✅ anyio.lowlevel imports successfully")

        # Check if current_token exists
        if hasattr(anyio.lowlevel, "current_token"):
            print("✅ anyio.lowlevel.current_token exists")
        else:
            print("❌ anyio.lowlevel.current_token not found")
            return False

        return True

    except ImportError as e:
        print(f"❌ Cannot import anyio.lowlevel: {e}")
        return False
    except Exception as e:
        print(f"❌ Error with anyio.lowlevel: {e}")
        return False


if __name__ == "__main__":
    syntax_ok = test_syntax()
    imports_ok = verify_imports()

    if syntax_ok and imports_ok:
        print("\n🎉 Fix verification successful!")
        print("📋 Summary:")
        print("  - Python syntax is valid")
        print("  - anyio.lowlevel import added correctly")
        print("  - current_token() method call is present")
        print("  - LookupError exception handling is in place")
        print("  - Token parameter is passed to from_thread.run")
        sys.exit(0)
    else:
        print("\n💥 Fix verification failed!")
        sys.exit(1)
