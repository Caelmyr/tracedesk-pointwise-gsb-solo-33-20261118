'use strict';
const ST={key:null,image:null,bitmap:null,stain:null,search:null,mode:'stain',preview:null,busy:false};
const ST_EDGE={top:'上边',right:'右边',bottom:'下边',left:'左边'};

function stCanvas(){return $('stainCanvas');}
function stBoxText(box){return box?box.map(v=>Math.round(v)).join(', '):'未设置';}
function stValidBox(box,minEdge){return Array.isArray(box)&&box.length===4&&box[2]-box[0]>=minEdge&&box[3]-box[1]>=minEdge;}

function stDraw(){
  const canvas=stCanvas();if(!canvas||!ST.bitmap)return;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(ST.bitmap,0,0);
  const rect=(box,color,fill,label)=>{
    if(!stValidBox(box,1))return;
    ctx.save();
    if(fill){ctx.fillStyle=fill;ctx.fillRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);}
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.strokeRect(box[0]+.5,box[1]+.5,box[2]-box[0]-1,box[3]-box[1]-1);
    ctx.font='12px system-ui,sans-serif';
    const width=ctx.measureText(label).width+8;
    const x=Math.max(1,Math.min(canvas.width-width-2,box[0]));
    const y=box[1]>=17?box[1]-16:box[1]+2;
    ctx.fillStyle=color;ctx.fillRect(x,y,width,15);
    ctx.fillStyle='#fffefa';ctx.fillText(label,x+4,y+12);
    ctx.restore();
  };
  rect(ST.search,'#c39a55','#c39a5533','候选范围');
  rect(ST.stain,'#d9502f','#d9502f40','污点框');
  if(ST.preview&&ST.preview.available&&ST.preview.analysis.patch)rect(ST.preview.analysis.patch.bounds,'#2f8b74',null,'补片');
}

function stSync(){
  const stain=$('stainBounds'),search=$('searchBounds');
  if(stain)stain.value=ST.stain?ST.stain.join(','):'';
  if(search)search.value=ST.search?ST.search.join(','):'';
  $('stainSummary').textContent='污点框 '+stBoxText(ST.stain)+' / 候选范围 '+stBoxText(ST.search);
  ['stain','search'].forEach(mode=>{$('stainMode_'+mode).classList.toggle('selected',ST.mode===mode);});
}

function stReadBox(id){
  const items=String($(id).value).split(/[,\s]+/).filter(Boolean).map(Number);
  if(items.length!==4||items.some(v=>!Number.isFinite(v)||v<0))return null;
  if(items[2]<=items[0]||items[3]<=items[1])return null;
  return items.map(v=>Math.round(v));
}

function stClearPreview(){ST.preview=null;const results=$('stainResults');if(results)results.innerHTML='';const state=$('stainPreviewState');if(state)state.textContent='尚未预览';stDraw();}

function stSetMode(mode){ST.mode=mode;stSync();}

function stPointer(event){
  const canvas=stCanvas(),bounds=canvas.getBoundingClientRect();
  if(!bounds.width||!bounds.height)return null;
  const x=(event.clientX-bounds.left)/bounds.width*canvas.width;
  const y=(event.clientY-bounds.top)/bounds.height*canvas.height;
  return [Math.max(0,Math.min(canvas.width-1,Math.round(x))),Math.max(0,Math.min(canvas.height-1,Math.round(y)))];
}

function stNormalise(a,b,mode){
  const width=ST.image.width,height=ST.image.height;
  let [x0,y0]=a,[x1,y1]=b;
  if(x1<x0)[x0,x1]=[x1,x0];
  if(y1<y0)[y0,y1]=[y1,y0];
  x1+=1;y1+=1;
  if(mode==='stain'){
    x0=Math.max(1,x0);y0=Math.max(1,y0);x1=Math.min(width-1,x1);y1=Math.min(height-1,y1);
    if(x1-x0>64)x1=x0+64;
    if(y1-y0>64)y1=y0+64;
  }else{x0=Math.max(0,x0);y0=Math.max(0,y0);x1=Math.min(width,x1);y1=Math.min(height,y1);}
  return [x0,y0,x1,y1];
}

function stBindCanvas(){
  const canvas=stCanvas();
  let anchor=null;
  canvas.onpointerdown=event=>{
    if(ST.busy)return;
    anchor=stPointer(event);
    if(!anchor)return;
    canvas.setPointerCapture(event.pointerId);
  };
  canvas.onpointermove=event=>{
    if(!anchor)return;
    const point=stPointer(event);
    if(!point)return;
    const box=stNormalise(anchor,point,ST.mode);
    if(ST.mode==='stain')ST.stain=box;else ST.search=box;
    ST.preview=null;
    stDraw();stSync();
  };
  canvas.onpointerup=()=>{
    anchor=null;
    const wide=ST.mode==='stain'?ST.stain:ST.search;
    if(wide&&(wide[2]-wide[0]<2||wide[3]-wide[1]<2)){
      if(ST.mode==='stain')ST.stain=null;else ST.search=null;
      toast('框选太小，请拖出一个区域');stDraw();stSync();
    }
  };
  canvas.onpointercancel=()=>{anchor=null;};
}

async function stLoadImage(key){
  const blob=await api('/projects/'+key+'/result-image/image');
  ST.bitmap=await createImageBitmap(blob);
  ST.image={width:ST.bitmap.width,height:ST.bitmap.height};
  const canvas=stCanvas();
  canvas.width=ST.image.width;canvas.height=ST.image.height;
  const dim=$('stainDimensions');
  if(dim)dim.textContent=ST.image.width+' × '+ST.image.height+' px / Y 向下';
  stDraw();
}

async function stUseSample(){
  const created=await api('/stains/sample','POST',{});
  await loadProject(created.project.id);
  closeDialog();
  await stainDialog({stain:created.stain,search:created.search});
}

async function stainDialog(initial){
  const projectMissing=!S.project||!S.project.current_result;
  openDialog('污点补片 · 预览与导出',`<div id="stainIntro" class="warning" ${projectMissing?'':'style="display:none"'}>当前工程还没有合成结果。可以先载入一张合成的旧织物示范图，或先在上方完成合成。</div>
  <div class="tabs">
    <button id="stainStart" class="primary">重载当前成图</button>
    <button id="stainSample" class="primary pale">载入旧织物示范</button>
    <button id="stainAuto">自动找污点</button>
    <button id="stainClear">清空两框</button>
  </div>
  <div class="row" style="margin-bottom:10px">
    <button id="stainMode_stain" class="selected">拖画污点框</button>
    <button id="stainMode_search">拖画候选范围</button>
    <span class="muted" id="stainDimensions">—</span>
  </div>
  <canvas id="stainCanvas" class="stain-canvas"></canvas>
  <div class="form-grid">
    <label class="field">污点框 [x0,y0,x1,y1]（右下不含）<input id="stainBounds" placeholder="例如 322,100,382,160"></label>
    <label class="field">候选范围 [x0,y0,x1,y1]<input id="searchBounds" placeholder="例如 200,0,494,280"></label>
  </div>
  <p class="muted" id="stainSummary">污点框 未设置 / 候选范围 未设置</p>
  <p class="muted">候选补片与污点框同尺寸；补片和它外面一像素的一圈都必须整个落在候选范围内，不碰污点框，也不取无效覆盖像素。四条边各自比较补片外一圈和污点外一圈同位置的平均 RGB 差，先保证最差的一边，再比四边平均。</p>
  <div class="tabs"><button id="stainPreviewBtn" class="primary">预览补片方案</button><button id="stainExportBtn">按此方案导出 PNG</button><span class="muted" id="stainPreviewState">尚未预览</span></div>
  <div id="stainResults"></div>`);
  ST.key=projectMissing?null:S.project.id;ST.stain=null;ST.search=null;ST.preview=null;ST.mode='stain';ST.bitmap=null;ST.image=null;
  bind('stainSample',stUseSample);
  bind('stainStart',async()=>{await stainDialog();});
  bind('stainClear',()=>{ST.stain=null;ST.search=null;ST.preview=null;stDraw();stSync();$('stainResults').innerHTML='';$('stainPreviewState').textContent='尚未预览';});
  bind('stainMode_stain',()=>stSetMode('stain'));
  bind('stainMode_search',()=>stSetMode('search'));
  bind('stainAuto',async()=>{
    if(!ST.key)throw Error('请先载入旧织物示范或合成当前工程');
    const result=await api('/projects/'+ST.key+'/stain/auto','POST',{});
    if(!result.detected){toast(result.reason||'没有自动找到污点');return;}
    ST.stain=result.stain;ST.search=result.search;ST.preview=null;
    stDraw();stSync();$('stainResults').innerHTML='';$('stainPreviewState').textContent='尚未预览';
    toast(result.note||'已按疑似污点预填，请确认框位');
  });
  bind('stainPreviewBtn',async()=>{await stPreview();});
  bind('stainExportBtn',async()=>{await stExport();});
  $('stainBounds').onchange=()=>{const box=stReadBox('stainBounds');if(box){ST.stain=box;ST.preview=null;stDraw();stSync();stClearPreview();}else toast('污点框需要四个整数坐标，且右下边界大于左上边界');};
  $('searchBounds').onchange=()=>{const box=stReadBox('searchBounds');if(box){ST.search=box;ST.preview=null;stDraw();stSync();stClearPreview();}else toast('候选范围需要四个整数坐标，且右下边界大于左上边界');};
  stBindCanvas();
  if(projectMissing||!S.project||!S.project.current_result){ST.key=null;stSync();return;}
  ST.key=S.project.id;
  try{
    await stLoadImage(ST.key);
  }catch(error){
    ST.key=null;
    const intro=$('stainIntro');
    intro.textContent='当前成图无法用于本次修补：'+error.message+' 可在“工程设置 → 工作长边”选512并重新合成，或先载入旧织物示范查看流程。';
    intro.style.display='block';
    stSync();
    return;
  }
  if(initial&&initial.stain){ST.stain=initial.stain;ST.search=initial.search||null;}
  stSync();
  if(initial&&initial.stain){try{await stPreview();}catch(error){toast(error.message);}}
  else if(S.project){try{const auto=await api('/projects/'+ST.key+'/stain/auto','POST',{});if(auto.detected){ST.stain=auto.stain;ST.search=auto.search;stDraw();stSync();}}catch(error){console.error(error);}}
}

function stEdgeTable(analysis){
  const rows=Object.keys(ST_EDGE).map(edge=>{
    const value=analysis.edge_differences_255[edge];
    const worst=analysis.worst_edge===edge;
    const channels=analysis.edge_channels[edge].channels_255;
    const width=Math.min(100,value*8).toFixed(1);
    return `<tr class="${worst?'worst':''}"><td>${ST_EDGE[edge]}</td><td>${value.toFixed(2)}</td><td>${channels.r.toFixed(1)}</td><td>${channels.g.toFixed(1)}</td><td>${channels.b.toFixed(1)}</td><td><span class="edge-bar" style="width:${width}%"></span></td></tr>`;
  }).join('');
  return `<table class="data-table stain-edges"><thead><tr><th>边</th><th>平均 RGB 差<br><small>0–255</small></th><th>R</th><th>G</th><th>B</th><th></th></tr></thead><tbody>${rows}</tbody></table>`;
}

async function stPreview(){
  if(!ST.key&&S.project&&S.project.current_result){ST.key=S.project.id;await stLoadImage(ST.key);}
  if(!ST.key)throw Error('请先载入旧织物示范或合成当前工程');
  if(!stValidBox(ST.stain,1))throw Error('请先框出要补的污点');
  if(ST.stain[2]-ST.stain[0]>64||ST.stain[3]-ST.stain[1]>64)throw Error('污点框单边不能超过64像素');
  if(ST.stain[0]<1||ST.stain[1]<1||ST.stain[2]>ST.image.width-1||ST.stain[3]>ST.image.height-1)throw Error('污点框要与成图边缘至少留1像素，才能比较框外一圈');
  ST.busy=true;
  try{
    const response=await api('/projects/'+ST.key+'/stain/preview','POST',{stain:ST.stain,search:ST.search});
    ST.preview=response;
    $('stainPreviewState').textContent=response.available?'方案已生成':'没有可用补片';
    stDraw();
    if(!response.available){
      const analysis=response.analysis||{};
      const suggestion=response.suggestion&&response.suggestion.detected?`<div class="row" style="margin-top:10px"><button id="stainUseSuggestion">采用自动框选 ${esc(response.suggestion.stain.join(', '))} ↗</button></div>`:'';
      $('stainResults').innerHTML=`<div class="warning">${esc(response.reason||'没有可用补片')}</div>${suggestion}<pre>${esc(JSON.stringify({候选范围内位置:analysis.positions_in_search,可用位置:analysis.candidate_count,排除:analysis.rejections},null,2))}</pre>`;
      const button=$('stainUseSuggestion');
      if(button)button.onclick=safe(()=>{ST.stain=response.suggestion.stain;ST.search=response.suggestion.search;stDraw();stSync();stPreview();});
      return;
    }
    const analysis=response.analysis,patch=analysis.patch;
    const verification=response.verification;
    $('stainResults').innerHTML=`<div class="stain-result-grid">
      <figure><img src="${response.overlay_png}" alt="补片位置与范围"><figcaption>用的是哪块：绿色为补片，红色为污点框，金色为候选范围</figcaption></figure>
      <figure><img class="pixelated" src="${response.patch_png}" alt="补片及外圈"><figcaption>补片与外一圈（放大显示）</figcaption></figure>
      <figure><img src="${response.focus_png}" alt="补完后局部"><figcaption>补完后局部</figcaption></figure>
      <figure><img src="${response.filled_png}" alt="补完后整张"><figcaption>补完的样子</figcaption></figure>
    </div>
    <div class="section-title">四边差异（补片外一圈对比污点外一圈，同位置平均 RGB）</div>
    ${stEdgeTable(analysis)}
    <div class="row" style="flex-wrap:wrap;margin-top:10px">
      <span class="count-chip">补片 ${patch.bounds.join(', ')}（${patch.width}×${patch.height}）</span>
      <span class="count-chip">最差边 ${analysis.worst_edge_label} ${analysis.worst_difference_255.toFixed(2)}</span>
      <span class="count-chip">四边平均 ${analysis.mean_difference_255.toFixed(2)}</span>
      <span class="count-chip">评估位置 ${analysis.candidate_count}</span>
      ${analysis.edges_good_except_worst?'<span class="count-chip">其余三边均明显更好，已优先压平最难边</span>':''}
    </div>
    <p class="muted">只替换污点框内 ${verification.stain_pixels} 个像素（实际变化 ${verification.changed_pixels} 个），框外变化 ${verification.changed_outside_stain} 个；补片按原方向原颜色复制，不旋转、不调色。原工程与已存合成结果不变。</p>
    <div class="section-title">次优候选（同样按最难边优先排序）</div>
    <table class="data-table"><thead><tr><th>补片位置</th><th>最难边差</th><th>四边平均差</th></tr></thead><tbody>${(analysis.alternatives||[]).map(row=>`<tr><td>${row.bounds.join(', ')}</td><td>${row.worst_difference_255.toFixed(2)}</td><td>${row.mean_difference_255.toFixed(2)}</td></tr>`).join('')||'<tr><td colspan="3">没有其他可用位置</td></tr>'}</tbody></table>`;
  }finally{ST.busy=false;}
}

async function stExport(){
  if(!ST.key&&S.project&&S.project.current_result){ST.key=S.project.id;await stLoadImage(ST.key);}
  if(!ST.key)throw Error('请先载入旧织物示范或合成当前工程');
  if(!ST.preview||!ST.preview.available)throw Error('请先预览，确认用的是哪块补片');
  const blob=await api('/projects/'+ST.key+'/stain/export','POST',{stain:ST.stain,search:ST.search,scheme_id:ST.preview.analysis.scheme_id,bits:8});
  downloadBlob(blob,'retouched.png');
  toast('已按预览方案导出 retouched.png；工程与原始成图未改动');
}
