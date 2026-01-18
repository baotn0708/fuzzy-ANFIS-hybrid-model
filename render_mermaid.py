#!/usr/bin/env python3
"""
Render Mermaid diagrams using Python
Alternative to mmdc CLI
"""

import subprocess
import os
from pathlib import Path

def render_with_playwright():
    """Use mermaid-py with playwright"""
    try:
        import mermaid as md
        from mermaid.graph import Graph
        
        diagram_dir = Path("diagrams")
        output_dir = diagram_dir / "rendered"
        output_dir.mkdir(exist_ok=True)
        
        print("🎨 Rendering Mermaid diagrams with mermaid.ink API...")
        
        for mmd_file in diagram_dir.glob("*.mmd"):
            print(f"  Rendering {mmd_file.name}...")
            
            with open(mmd_file, 'r', encoding='utf-8') as f:
                diagram_text = f.read()
            
            # Use mermaid.ink API
            output_file = output_dir / f"{mmd_file.stem}.png"
            
            # Create URL encoded version
            import urllib.parse
            import base64
            
            # Encode diagram
            encoded = base64.urlsafe_b64encode(diagram_text.encode('utf-8')).decode('ascii')
            url = f"https://mermaid.ink/img/{encoded}"
            
            # Download image
            import urllib.request
            try:
                urllib.request.urlretrieve(url, output_file)
                print(f"    ✅ {output_file.name}")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        print(f"\n✅ Done! Check {output_dir}/")
        
    except ImportError:
        print("❌ mermaid package not found")
        return False
    
    return True

def render_with_cli():
    """Try using mmdc CLI if available"""
    try:
        result = subprocess.run(['which', 'mmdc'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ mmdc not found in PATH")
            return False
        
        mmdc_path = result.stdout.strip()
        print(f"✅ Found mmdc at: {mmdc_path}")
        
        diagram_dir = Path("diagrams")
        output_dir = diagram_dir / "rendered"
        output_dir.mkdir(exist_ok=True)
        
        for mmd_file in diagram_dir.glob("*.mmd"):
            output_file = output_dir / f"{mmd_file.stem}.png"
            print(f"  Rendering {mmd_file.name}...")
            
            cmd = [
                'mmdc',
                '-i', str(mmd_file),
                '-o', str(output_file),
                '-b', 'transparent',
                '-w', '1920',
                '-H', '1080'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"    ✅ {output_file.name}")
            else:
                print(f"    ❌ Error: {result.stderr}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with mmdc: {e}")
        return False

if __name__ == '__main__':
    print("="*60)
    print("🎨 RENDERING MERMAID DIAGRAMS")
    print("="*60 + "\n")
    
    # Try API method first (fastest, no installation needed)
    if render_with_playwright():
        print("\n✨ Success using mermaid.ink API!")
    elif render_with_cli():
        print("\n✨ Success using mmdc CLI!")
    else:
        print("\n⚠️  Both methods failed. Please install:")
        print("   npm install -g @mermaid-js/mermaid-cli")
        print("   or")
        print("   pip install mermaid-py")
