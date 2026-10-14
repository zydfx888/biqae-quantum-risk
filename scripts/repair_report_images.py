"""Use portable Markdown images and build self-contained HTML without dependencies."""
from pathlib import Path
import re
import json
import hashlib
import html
import base64
import shutil
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
MD=ROOT/'综合实验报告.md'
DEST=ROOT/'综合实验报告.html'
ASSETS=ROOT/'report_assets'
ASSETS.mkdir(exist_ok=True)
text=MD.read_text(encoding='utf-8')
records=[]

def fix_image(m):
    label,target=m.groups()
    source=Path(target)
    if not source.is_absolute():source=ROOT/source
    with Image.open(source) as im:im.verify()
    dest=ASSETS/f'figure_{len(records)+1:02d}.png'
    if source.resolve()!=dest.resolve():shutil.copyfile(source,dest)
    records.append({'label':label,'source':str(source),'relative':dest.relative_to(ROOT).as_posix(),
                    'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    return f'![{label}]({dest.relative_to(ROOT).as_posix()})'

# Restore the main binning figure that had become an ordinary image-file link.
text=re.sub(r'!?\[([^\]\n]+)\]\(([^\)\n]+\.(?:png|jpg|jpeg))\)',fix_image,text)
MD.write_text(text,encoding='utf-8')

def inline(s):
    chunks=[]
    def hold(value):
        chunks.append(value);return f'\x01{len(chunks)-1}\x02'
    s=re.sub(r'`([^`]+)`',lambda m:hold('<code>'+html.escape(m[1])+'</code>'),s)
    def link(m):
        label,url=m.groups()
        if re.match(r'^[A-Za-z]:[/\\]',url):url=Path(url).as_uri()
        return hold('<a href="'+html.escape(url,quote=True)+'">'+html.escape(label)+'</a>')
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,s)
    s=html.escape(s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
    for i,value in enumerate(chunks):s=s.replace(f'\x01{i}\x02',value)
    return s

lines=text.splitlines();body=[];i=0;image_count=0
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('```'):
        block=[];i+=1
        while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
        body.append('<pre><code>'+html.escape('\n'.join(block))+'</code></pre>');i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cells=lines[i].strip().strip('|').split('|');i+=1
            if all(re.fullmatch(r'\s*:?-+:?\s*',c) for c in cells):continue
            tag='th' if not rows else 'td'
            rows.append('<tr>'+''.join(f'<{tag}>'+inline(c.strip())+f'</{tag}>' for c in cells)+'</tr>')
        body.append('<div class="table-wrap"><table>'+''.join(rows)+'</table></div>');continue
    image=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
    if image:
        p=ROOT/image[2];encoded=base64.b64encode(p.read_bytes()).decode('ascii')
        body.append('<figure><img alt="'+html.escape(image[1],quote=True)+'" src="data:image/png;base64,'+encoded+'"><figcaption>'+html.escape(image[1])+'</figcaption></figure>')
        image_count+=1;i+=1;continue
    if re.fullmatch(r'<a id="[^"]+"></a>',line):body.append(line);i+=1;continue
    heading=re.match(r'^(#{1,6})\s+(.+)',line)
    if heading:
        n=len(heading[1]);body.append(f'<h{n}>'+inline(heading[2])+f'</h{n}>');i+=1;continue
    if line.startswith(('- ','* ')):
        items=[]
        while i<len(lines) and lines[i].strip().startswith(('- ','* ')):
            items.append('<li>'+inline(lines[i].strip()[2:])+'</li>');i+=1
        body.append('<ul>'+''.join(items)+'</ul>');continue
    if re.match(r'^\d+\. ',line):
        items=[]
        while i<len(lines) and re.match(r'^\d+\. ',lines[i].strip()):
            items.append('<li>'+inline(re.sub(r'^\d+\. ','',lines[i].strip()))+'</li>');i+=1
        body.append('<ol>'+''.join(items)+'</ol>');continue
    if line.startswith('> '):body.append('<blockquote>'+inline(line[2:])+'</blockquote>');i+=1;continue
    body.append('<p>'+inline(line)+'</p>');i+=1
page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>方案三：全部实验结果综合报告</title><style>
body{max-width:1200px;margin:36px auto;padding:0 28px;font:16px/1.8 "Microsoft YaHei",sans-serif;color:#203047;background:#fff}h1{font-size:30px}h2{margin-top:42px;border-bottom:2px solid #dce5ed;padding-bottom:8px}h3{margin-top:28px}a{color:#086b93}p,li{overflow-wrap:anywhere}.table-wrap{overflow-x:auto}table{border-collapse:collapse;min-width:70%;font-size:14px;margin:18px 0}td,th{border:1px solid #d8e1eb;padding:8px 12px;text-align:left}th{background:#edf4f8}figure{margin:24px 0}img{display:block;max-width:100%;height:auto;margin:auto}figcaption{text-align:center;color:#65748b;font-size:14px}blockquote{border-left:4px solid #d2a35c;background:#fff7e9;padding:10px 16px;margin:16px 0}pre{background:#f3f6f9;padding:16px;overflow:auto}code{font-size:14px}a[id]{scroll-margin-top:20px}@media print{body{max-width:none;margin:0;font-size:11pt}table{font-size:9pt}tr,figure{break-inside:avoid}}
</style></head><body>'''+ '\n'.join(body)+'</body></html>'
DEST.write_text(page,encoding='utf-8')
assert image_count==len(records) and image_count>0
assert len(re.findall(r'<img[^>]+src="data:image/png;base64,',page))==image_count
for encoded in re.findall(r'<img[^>]+src="data:image/png;base64,([^"]+)"',page):
    import io
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as im:im.verify()
meta=ROOT/'artifacts/report_consolidation'
(meta/'image_repair.json').write_text(json.dumps({'markdown_sha256':hashlib.sha256(MD.read_bytes()).hexdigest(),
    'html_sha256':hashlib.sha256(DEST.read_bytes()).hexdigest(),'verified_images':image_count,'images':records},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'images_verified':image_count,'html_bytes':DEST.stat().st_size,'html':str(DEST)},ensure_ascii=False))
