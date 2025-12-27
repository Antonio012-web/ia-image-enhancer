// static/script.js — preview enhance + vectorización inline (SVG/PDF) + MODAL Antes/Después
document.addEventListener('DOMContentLoaded', () => {
  const fileInput   = document.getElementById('fileInput');
  const dropzone    = document.getElementById('dropzone');

  // ====== Enhance controls ======
  const scaleSel    = document.getElementById('scale');
  const denoiseR    = document.getElementById('denoise');
  const sharpenR    = document.getElementById('sharpen');
  const denoiseVal  = document.getElementById('denoiseVal');
  const sharpenVal  = document.getElementById('sharpenVal');
  const formatSel   = document.getElementById('format');
  const presetSel   = document.getElementById('preset');
  const btnProcess  = document.getElementById('btnProcess');
  const btnDownload = document.getElementById('btnDownload');

  const detailLevel = document.getElementById('detailLevel');
  const detailVal   = document.getElementById('detailVal');
  const textMode    = document.getElementById('textMode');
  const textControls= document.getElementById('textControls');
  const claheClip   = document.getElementById('claheClip');
  const claheTile   = document.getElementById('claheTile');
  const guidedRadius= document.getElementById('guidedRadius');
  const guidedEps   = document.getElementById('guidedEps');
  const textSharpen = document.getElementById('textSharpen');
  const lapAmount   = document.getElementById('lapAmount');
  const finalDownscale = document.getElementById('finalDownscale');

  const deblurChk   = document.getElementById('deblur');
  const deblurRadius= document.getElementById('deblurRadius');
  const deblurLambda= document.getElementById('deblurLambda');
  const msAmount    = document.getElementById('msAmount');

  const nlmInput    = document.getElementById('nlm');
  const nlmChroma   = document.getElementById('nlmChroma');
  const antiring    = document.getElementById('antiring');

  const edgeclip    = document.getElementById('edgeclip');
  const edgeclipAmt = document.getElementById('edgeclipAmt');
  const edgeclipClip= document.getElementById('edgeclipClip');
  const paperSmooth = document.getElementById('paperSmooth');

  // lienzos mejorar
  const cOrg  = document.getElementById('canvasOriginal');
  const cRes  = document.getElementById('canvasResult');
  const ctxOrg= cOrg ? cOrg.getContext('2d') : null;
  const ctxRes= cRes ? cRes.getContext('2d') : null;

  const detailSpark = document.getElementById('detailSpark');
  const barHigh = document.getElementById('barHigh');
  const barMid  = document.getElementById('barMid');
  const barLow  = document.getElementById('barLow');
  const barEdge = document.getElementById('barEdge');
  const valHigh = document.getElementById('valHigh');
  const valMid  = document.getElementById('valMid');
  const valLow  = document.getElementById('valLow');
  const valEdge = document.getElementById('valEdge');

  // ====== Vectorize controls ======
  const btnVectorize= document.getElementById('btnVectorize');
  const vecMode     = document.getElementById('vecMode');
  const vecColors   = document.getElementById('vecColors');
  const vecSmooth   = document.getElementById('vecSmooth');
  const vecMinArea  = document.getElementById('vecMinArea');
  const vecFormat   = document.getElementById('vecFormat');
  const fastMode    = document.getElementById('fastMode');
  const vecDownload = document.getElementById('vecDownload');
  const vecPreview  = document.getElementById('vecPreview');

  // lienzos vectorización
  const cVecOrg  = document.getElementById('canvasVecOriginal');
  const cVecRes  = document.getElementById('canvasVecResult');
  const ctxVecOrg= cVecOrg ? cVecOrg.getContext('2d') : null;
  const ctxVecRes= cVecRes ? cVecRes.getContext('2d') : null;

  // cajas de preview entrada
  const btnUpload = document.getElementById('btnUpload');

  const enhInputBox   = document.getElementById('enhInputBox');
  const enhInputThumb = document.getElementById('enhInputThumb');
  const enhInputName  = document.getElementById('enhInputName');
  const enhInputMeta  = document.getElementById('enhInputMeta');
  const enhPick       = document.getElementById('enhPick');

  const vecInputBox   = document.getElementById('vecInputBox');
  const vecInputThumb = document.getElementById('vecInputThumb');
  const vecInputName  = document.getElementById('vecInputName');
  const vecInputMeta  = document.getElementById('vecInputMeta');
  const vecPick       = document.getElementById('vecPick');

  let originalImage = null;
  let resultImage   = null;
  let lastPreviewController = null;

  // ====== Utils ======
  function fitCanvas(canvas, img){
    const maxW = 600;
    const ratio = Math.min(maxW / img.width, 1);
    canvas.width  = Math.round(img.width * ratio);
    canvas.height = Math.round(img.height * ratio);
  }
  function drawToCanvas(canvas, ctx, img){
    if (!canvas || !ctx || !img) return;
    fitCanvas(canvas, img);
    ctx.clearRect(0,0,canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  }
  function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }
  const debounce = (fn, ms=350) => { let t; return (...args)=>{ clearTimeout(t); t = setTimeout(()=>fn(...args), ms); }; };

  function formatBytes(bytes){
    if (!Number.isFinite(bytes)) return '—';
    const units = ['B','KB','MB','GB'];
    let u = 0, val = bytes;
    while (val >= 1024 && u < units.length-1){ val/=1024; u++; }
    return `${val.toFixed(val<10?1:0)} ${units[u]}`;
  }

  // ====== Carga de archivo ======
  function loadFile(file){
    const reader = new FileReader();
    reader.onload = e => {
      const img = new Image();
      img.onload = () => {
        originalImage = img;

        // Lienzos original
        drawToCanvas(cOrg,    ctxOrg,    img);
        drawToCanvas(cVecOrg, ctxVecOrg, img);

        // Previews de entrada
        try {
          const url = URL.createObjectURL(file);

          if (enhInputThumb){ enhInputThumb.src = url; enhInputThumb.style.display = 'block'; }
          if (enhInputName)  enhInputName.textContent = file.name || '(imagen)';
          if (enhInputMeta)  enhInputMeta.textContent = `${img.width}×${img.height} • ${formatBytes(file.size)}`;
          if (enhInputBox)   enhInputBox.style.opacity = '1';

          if (vecInputThumb){ vecInputThumb.src = url; vecInputThumb.style.display = 'block'; }
          if (vecInputName)  vecInputName.textContent = file.name || '(imagen)';
          if (vecInputMeta)  vecInputMeta.textContent = `${img.width}×${img.height} • ${formatBytes(file.size)}`;
          if (vecInputBox)   vecInputBox.style.opacity = '1';
        } catch (err) {
          console.warn('Error actualizando previews de entrada:', err);
        }
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  fileInput?.addEventListener('change', e => {
    if (e.target.files?.[0]) loadFile(e.target.files[0]);
  });
  if (dropzone){
    dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('hover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('hover'));
    dropzone.addEventListener('drop', e => {
      e.preventDefault(); dropzone.classList.remove('hover');
      const file = e.dataTransfer.files[0];
      if (file) loadFile(file);
    });
  }

  btnUpload?.addEventListener('click', () => fileInput?.click());
  enhPick  ?.addEventListener('click', () => fileInput?.click());
  vecPick  ?.addEventListener('click', () => fileInput?.click());

  // ====== Labels ======
  const updateLabels = () => {
    if (denoiseVal && denoiseR) denoiseVal.textContent = denoiseR.value;
    if (sharpenVal && sharpenR) sharpenVal.textContent = parseFloat(sharpenR.value).toFixed(1);
    if (detailVal && detailLevel) detailVal.textContent = detailLevel.value;
  };
  ['input','change'].forEach(ev => {
    denoiseR?.addEventListener(ev, updateLabels);
    sharpenR?.addEventListener(ev, updateLabels);
    detailLevel?.addEventListener(ev, () => { updateLabels(); applyDetail(detailLevel.value); previewEnhanceDebounced(); });
  });
  updateLabels();

  if (textMode && textControls){
    textMode.addEventListener('change', () => {
      textControls.style.display = textMode.checked ? 'block' : 'none';
      previewEnhanceDebounced();
    });
  }

  // ====== Presets ======
  function applyPreset(name){
    if (name === 'none'){
      textMode.checked = false; textControls.style.display = 'none';
      deblurChk.checked = false;
      scaleSel.value = '1'; denoiseR.value = '0'; sharpenR.value = '1.0'; formatSel.value = 'png';
      msAmount.value = '0.4'; nlmInput.value = '0'; nlmChroma.checked = false; antiring.value = '0.0';
      edgeclip.checked = false; edgeclipAmt.value = '0.55'; edgeclipClip.value = '0.015'; paperSmooth.value = '0.0';
      updateLabels(); previewEnhanceDebounced(); return;
    }
    textControls.style.display = 'block';
    switch(name){
      case 'doc_texto':
        textMode.checked = true; deblurChk.checked = true;
        scaleSel.value = '4'; finalDownscale.value = '0.5';
        claheClip.value = '3.0'; claheTile.value = '8';
        guidedRadius.value = '8'; guidedEps.value = '0.000001';
        textSharpen.value = '0.5'; lapAmount.value = '0.22';
        deblurRadius.value = '1.5'; deblurLambda.value = '0.01'; msAmount.value = '0.5';
        nlmInput.value = '12'; nlmChroma.checked = true; antiring.value = '0.25';
        edgeclip.checked = true; edgeclipAmt.value = '0.55'; edgeclipClip.value = '0.015'; paperSmooth.value = '0.25';
        denoiseR.value = '0'; sharpenR.value = '0.8'; formatSel.value = 'png';
        break;
      case 'ui_screen':
        textMode.checked = true; deblurChk.checked = false;
        scaleSel.value = '2'; finalDownscale.value = '1';
        claheClip.value = '2.2'; claheTile.value = '8';
        guidedRadius.value = '5'; guidedEps.value = '0.000001';
        textSharpen.value = '0.5'; lapAmount.value = '0.20';
        deblurRadius.value = '1.2'; deblurLambda.value = '0.01'; msAmount.value = '0.45';
        nlmInput.value = '8'; nlmChroma.checked = true; antiring.value = '0.2';
        edgeclip.checked = true; edgeclipAmt.value = '0.5'; edgeclipClip.value = '0.018'; paperSmooth.value = '0.15';
        denoiseR.value = '0'; sharpenR.value = '0.8'; formatSel.value = 'png';
        break;
      case 'foto_texto':
        textMode.checked = true; deblurChk.checked = true;
        scaleSel.value = '4'; finalDownscale.value = '0.75';
        claheClip.value = '3.5'; claheTile.value = '10';
        guidedRadius.value = '7'; guidedEps.value = '0.000003';
        textSharpen.value = '0.6'; lapAmount.value = '0.28';
        deblurRadius.value = '1.8'; deblurLambda.value = '0.012'; msAmount.value = '0.55';
        nlmInput.value = '14'; nlmChroma.checked = true; antiring.value = '0.3';
        edgeclip.checked = true; edgeclipAmt.value = '0.55'; edgeclipClip.value = '0.016'; paperSmooth.value = '0.2';
        denoiseR.value = '10'; sharpenR.value = '1.0'; formatSel.value = 'png';
        break;
      case 'microtext':
        textMode.checked = false; deblurChk.checked = true;
        scaleSel.value = '4'; finalDownscale.value = '0.5';
        claheClip.value = '2.6'; claheTile.value = '8';
        guidedRadius.value = '8'; guidedEps.value = '0.000002';
        textSharpen.value = '0.6'; lapAmount.value = '0.18';
        deblurRadius.value = '1.4'; deblurLambda.value = '0.012'; msAmount.value = '0.35';
        nlmInput.value = '12'; nlmChroma.checked = true; antiring.value = '0.35';
        edgeclip.checked = true; edgeclipAmt.value = '0.55'; edgeclipClip.value = '0.015';
        paperSmooth.value = '0.25';
        denoiseR.value = '0'; sharpenR.value = '0.8'; formatSel.value = 'png';
        break;
    }
    updateLabels(); previewEnhanceDebounced();
  }
  presetSel?.addEventListener('change', () => applyPreset(presetSel.value));

  // ====== Mapeo del nivel de detalle ======
  function applyDetail(vRaw){
    const v = clamp(parseInt(vRaw || 0, 10), 0, 100);
    const ms   = 0.20 + 0.006 * v;
    const usm  = 0.50 + 0.008 * v;
    const lap  = 0.10 + 0.003 * v;
    const eAmt = 0.45 + 0.004 * v;
    const eClip= 0.015;
    const nlm  = Math.round(16 - 0.10 * v);
    const ar   = 0.20 + 0.005 * v;
    const dRad = 1.20 + 0.006 * v;

    msAmount.value     = ms.toFixed(2);
    textSharpen.value  = usm.toFixed(2);
    lapAmount.value    = lap.toFixed(2);
    edgeclip.checked   = true;
    edgeclipAmt.value  = eAmt.toFixed(2);
    edgeclipClip.value = eClip.toString();
    nlmInput.value     = clamp(nlm, 6, 20);
    antiring.value     = ar.toFixed(2);
    deblurChk.checked  = true;
    deblurRadius.value = dRad.toFixed(2);
    deblurLambda.value = '0.010';
    sharpenR.value     = (0.8 + 0.004 * v).toFixed(2);

    if (textMode?.checked){
      claheClip.value    = (2.2 + 0.006 * v).toFixed(2);
      guidedRadius.value = Math.round(clamp(5 + v/25, 5, 9));
    }
    updateLabels();
  }

  const previewEnhanceDebounced = debounce(previewEnhance, 400);

  // ====== Vista previa rápida (Mejorar) ======
  async function previewEnhance(){
    try{
      if (!originalImage) return;
      if (!fileInput?.files?.[0]) return;

      if (lastPreviewController) lastPreviewController.abort();
      const controller = new AbortController();
      lastPreviewController = controller;

      const maxW = 1024;
      const ratio = Math.min(maxW / originalImage.width, 1);
      const pw = Math.round(originalImage.width * ratio);
      const ph = Math.round(originalImage.height * ratio);
      const off = document.createElement('canvas');
      off.width = pw; off.height = ph;
      const octx = off.getContext('2d');
      octx.imageSmoothingQuality = 'high';
      octx.drawImage(originalImage, 0, 0, pw, ph);
      const blob = await new Promise(res => off.toBlob(res, 'image/png', 1.0));
      if (!blob) return;

      const fd = new FormData();
      fd.append('image', blob, 'preview.png');
      fd.append('format', 'png');
      fd.append('scale',   scaleSel?.value ?? '1');
      fd.append('denoise', denoiseR?.value ?? '0');
      fd.append('sharpen', sharpenR?.value ?? '1.0');
      fd.append('deblur', deblurChk?.checked ? '1' : '0');
      fd.append('deblur_radius', deblurRadius?.value ?? '1.5');
      fd.append('deblur_lambda', deblurLambda?.value ?? '0.01');
      fd.append('ms_amount', msAmount?.value ?? '0.5');
      fd.append('nlm', nlmInput?.value ?? '12');
      fd.append('nlm_chroma', nlmChroma?.checked ? '1' : '0');
      fd.append('antiring', antiring?.value ?? '0.25');
      fd.append('edgeclip', edgeclip?.checked ? '1' : '0');
      fd.append('edgeclip_amt', edgeclipAmt?.value ?? '0.55');
      fd.append('edgeclip_clip', edgeclipClip?.value ?? '0.015');
      fd.append('paper_smooth', paperSmooth?.value ?? '0.25');

      const isMicro = presetSel?.value === 'microtext';
      if (isMicro){
        fd.append('microtext_mode', '1');
        fd.append('clahe_clip', claheClip?.value ?? '2.6');
        fd.append('clahe_tile', claheTile?.value ?? '8');
        fd.append('guided_radius', guidedRadius?.value ?? '8');
        fd.append('guided_eps', guidedEps?.value ?? '0.000002');
        fd.append('text_sharpen', textSharpen?.value ?? '0.6');
        fd.append('lap_amount', lapAmount?.value ?? '0.18');
        fd.append('final_downscale', finalDownscale?.value ?? '0.5');
      } else if (textMode?.checked){
        fd.append('text_mode', '1');
        fd.append('clahe_clip', claheClip?.value ?? '2.5');
        fd.append('clahe_tile', claheTile?.value ?? '8');
        fd.append('guided_radius', guidedRadius?.value ?? '6');
        fd.append('guided_eps', guidedEps?.value ?? '0.000001');
        fd.append('text_sharpen', textSharpen?.value ?? '0.6');
        fd.append('lap_amount', lapAmount?.value ?? '0.25');
        fd.append('final_downscale', finalDownscale?.value ?? '1');
      } else {
        fd.append('text_mode', '0');
        fd.append('microtext_mode', '0');
      }

      btnProcess.textContent = 'Vista previa…';
      btnProcess.disabled = true;

      const res = await fetch('/api/enhance', { method:'POST', body: fd, signal: controller.signal });
      const ctype = res.headers.get('content-type') || '';
      const raw = await res.text();
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${raw.slice(0,200)}`);
      if (!ctype.includes('application/json')) throw new Error(`Respuesta no JSON: ${raw.slice(0,200)}`);
      const data = JSON.parse(raw);

      const viewUrl = data.view_url || data.url;
      const imgRes = await fetch(viewUrl, { signal: controller.signal });
      const imgBlob = await imgRes.blob();
      const img = new Image();
      img.onload = () => {
        resultImage = img;
        drawToCanvas(cRes, ctxRes, img);
        setTimeout(() => computeDetailLevels(), 0);
      };
      img.src = URL.createObjectURL(imgBlob);
    } catch (e){
      if (e.name !== 'AbortError') console.error('[preview] error', e);
    } finally {
      btnProcess.textContent = 'Mejorar';
      btnProcess.disabled = false;
    }
  }

  // ====== Proceso completo (Mejorar) ======
  btnProcess?.addEventListener('click', async () => {
    if (!fileInput?.files?.[0]) { alert('Selecciona una imagen primero.'); return; }
    btnProcess.disabled = true; btnProcess.textContent = 'Procesando...';
    btnDownload.style.display = 'none';

    const fd = new FormData();
    fd.append('image', fileInput.files[0]);
    fd.append('scale',   scaleSel?.value ?? '1');
    fd.append('denoise', denoiseR?.value ?? '0');
    fd.append('sharpen', sharpenR?.value ?? '1.0');
    fd.append('format',  formatSel?.value ?? 'png');
    fd.append('deblur', deblurChk?.checked ? '1' : '0');
    fd.append('deblur_radius', deblurRadius?.value ?? '1.5');
    fd.append('deblur_lambda', deblurLambda?.value ?? '0.01');
    fd.append('ms_amount', msAmount?.value ?? '0.5');
    fd.append('nlm', nlmInput?.value ?? '12');
    fd.append('nlm_chroma', nlmChroma?.checked ? '1' : '0');
    fd.append('antiring', antiring?.value ?? '0.25');
    fd.append('edgeclip', edgeclip?.checked ? '1' : '0');
    fd.append('edgeclip_amt', edgeclipAmt?.value ?? '0.55');
    fd.append('edgeclip_clip', edgeclipClip?.value ?? '0.015');
    fd.append('paper_smooth', paperSmooth?.value ?? '0.25');

    const isMicrotext = (presetSel?.value === 'microtext');
    if (isMicrotext){
      fd.append('microtext_mode', '1');
      fd.append('clahe_clip', claheClip?.value ?? '2.6');
      fd.append('clahe_tile', claheTile?.value ?? '8');
      fd.append('guided_radius', guidedRadius?.value ?? '8');
      fd.append('guided_eps', guidedEps?.value ?? '0.000002');
      fd.append('text_sharpen', textSharpen?.value ?? '0.6');
      fd.append('lap_amount', lapAmount?.value ?? '0.18');
      fd.append('final_downscale', finalDownscale?.value ?? '0.5');
    } else if (textMode?.checked){
      fd.append('text_mode', '1');
      fd.append('clahe_clip', claheClip?.value ?? '2.5');
      fd.append('clahe_tile', claheTile?.value ?? '8');
      fd.append('guided_radius', guidedRadius?.value ?? '6');
      fd.append('guided_eps', guidedEps?.value ?? '0.000001');
      fd.append('text_sharpen', textSharpen?.value ?? '0.6');
      fd.append('lap_amount', lapAmount?.value ?? '0.25');
      fd.append('final_downscale', finalDownscale?.value ?? '1');
    } else {
      fd.append('text_mode', '0');
      fd.append('microtext_mode', '0');
    }

    try{
      const res = await fetch('/api/enhance', { method: 'POST', body: fd });
      const ctype = res.headers.get('content-type') || '';
      const raw = await res.text();
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${raw.slice(0,200)}`);
      if (!ctype.includes('application/json')) throw new Error(`Respuesta no JSON: ${raw.slice(0,200)}`);
      const data = JSON.parse(raw);

      const viewUrl = data.view_url || data.url;
      const downloadUrl = data.download_url || data.url;

      const fileRes = await fetch(viewUrl);
      const blob = await fileRes.blob();

      if (data.mime?.startsWith('image/')){
        const img = new Image();
        img.onload = () => {
          resultImage = img;
          drawToCanvas(cRes, ctxRes, img);
          setTimeout(() => computeDetailLevels(), 0);
        };
        img.src = URL.createObjectURL(blob);
      } else {
        if (cRes) { cRes.width = cRes.height = 0; }
      }

      btnDownload.href = downloadUrl;
      btnDownload.download = data.filename;
      btnDownload.style.display = 'inline-block';

      await openCompareModal({
        url: viewUrl,
        filename: data.filename,
        mime: data.mime
      });

    } catch(err){
      console.error(err);
      alert('Error: ' + err.message);
    } finally {
      btnProcess.disabled = false; btnProcess.textContent = 'Mejorar';
    }
  });

  // ====== Vectorización ======
  btnVectorize?.addEventListener('click', async () => {
    try{
      let blobToSend = null;
      if (fileInput?.files?.[0]) {
        blobToSend = fileInput.files[0];
      } else if (cOrg && cOrg.width > 0 && cOrg.height > 0) {
        blobToSend = await new Promise(res => cOrg.toBlob(res, 'image/png', 1.0));
      }
      if (!blobToSend) { alert('Carga primero una imagen para vectorizar.'); return; }

      const fd = new FormData();
      fd.append('image', blobToSend, blobToSend.name || 'image.png');
      fd.append('mode', (vecMode?.value || 'mono'));
      fd.append('colors', (vecColors?.value || '6'));
      fd.append('svg_smooth', (vecSmooth?.value || '2.0'));
      fd.append('min_area', (vecMinArea?.value || '25'));
      fd.append('format', (vecFormat?.value || 'svg'));

      if (fastMode?.checked) {
        fd.append('pro', '1');
        fd.append('pro_upsample', '2');
        fd.append('pro_smooth', '1.0');
        fd.append('pro_edge', '0.6');
      }

      btnVectorize.disabled = true;
      btnVectorize.textContent = 'Vectorizando…';
      vecDownload.style.display = 'none';

      const res = await fetch('/api/vectorize', { method: 'POST', body: fd });
      const ctype = res.headers.get('content-type') || '';
      const raw = await res.text();
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${raw.slice(0,200)}`);
      if (!ctype.includes('application/json')) throw new Error(`Respuesta no JSON: ${raw.slice(0,200)}`);
      const data = JSON.parse(raw);

      const viewUrl = data.view_url || data.url;
      const downloadUrl = data.download_url || data.url;

      renderVectorResult({
        url: viewUrl,
        filename: data.filename,
        mime: data.mime
      });

      // DIBUJAR RESULTADO EN LIENZO "Vectorizado"
      if (data.mime && data.mime.startsWith('image/')) {
        const img = new Image();
        img.onload = () => drawToCanvas(cVecRes, ctxVecRes, img);
        img.src = viewUrl;
      } else if (data.mime && data.mime.includes('svg')) {
        const img = new Image();
        img.onload = () => drawToCanvas(cVecRes, ctxVecRes, img);
        img.src = viewUrl;
      } else {
        if (cVecRes) { cVecRes.width = cVecRes.height = 0; }
      }

      await openCompareModal({
        url: viewUrl,
        filename: data.filename,
        mime: data.mime
      });

      vecDownload.href = downloadUrl;
      vecDownload.download = data.filename || 'vectorizado';
      vecDownload.style.display = 'inline-block';

    } catch (err) {
      console.error('[vectorize] error', err);
      alert('Error al vectorizar: ' + err.message);
    } finally {
      btnVectorize.disabled = false;
      btnVectorize.textContent = 'Vectorizar';
    }
  });

  // Inserta resultado en #vecPreview (SVG/PDF)
  function renderVectorResult({ url, filename, mime }){
    if (!vecPreview) return;
    vecPreview.innerHTML = '';

    if (mime === 'application/pdf') {
      const ifr = document.createElement('iframe');
      ifr.src = url;
      ifr.title = filename || 'vectorizado.pdf';
      ifr.style.cssText = 'width:100%; height:240px; background:white; border:0;';
      vecPreview.appendChild(ifr);
    } else {
      const obj = document.createElement('object');
      obj.type = 'image/svg+xml';
      obj.data = url;
      obj.title = filename || 'vectorizado.svg';
      obj.style.cssText = 'width:100%; min-height:240px; border:0;';
      vecPreview.appendChild(obj);
    }
  }

  // ====== Modal Antes / Después ======
  function openCompareModal({ url, filename, mime }) {
    return new Promise(async (resolve) => {
      const overlay = document.createElement('div');
      overlay.style.cssText = `position:fixed; inset:0; background:rgba(0,0,0,0.55); display:flex; align-items:center; justify-content:center; z-index:9999; padding:20px;`;
      const modal = document.createElement('div');
      modal.style.cssText = `width: min(1100px, 95vw); max-height: 90vh; overflow:auto; background:#0b1220; border:1px solid rgba(255,255,255,0.12); border-radius:12px; box-shadow:0 10px 40px rgba(0,0,0,0.5); padding:16px;`;
      overlay.appendChild(modal);
      modal.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:12px;">
          <h3 style="margin:0;">Vista previa — Antes / Después</h3>
          <button id="vpClose" class="btn" style="background:#1f2937;">Cerrar</button>
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
          <div>
            <div style="font-size:.9rem; opacity:.8; margin-bottom:6px;">Antes (bitmap)</div>
            <div style="border:1px solid rgba(255,255,255,0.12); border-radius:10px; padding:10px; overflow:auto;">
              <canvas id="vpOriginal" style="max-width:100%; height:auto; display:block;"></canvas>
            </div>
          </div>
          <div>
            <div style="font-size:.9rem; opacity:.8; margin-bottom:6px;">Después (${mime === 'application/pdf' ? 'PDF' : (mime?.includes('svg') ? 'SVG' : 'Imagen')})</div>
            <div id="vpAfterWrap" style="border:1px solid rgba(255,255,255,0.12); border-radius:10px; padding:10px; height:520px; overflow:auto;"></div>
          </div>
        </div>
        <div style="display:flex; justify-content:flex-end; gap:8px; margin-top:14px;">
          <a id="vpDownload" class="btn" href="#" download>Descargar</a>
          <button id="vpCancel" class="btn" style="background:#1f2937;">Cancelar</button>
        </div>
      `;
      document.body.appendChild(overlay);

      const vpOriginal = modal.querySelector('#vpOriginal');
      if (vpOriginal && cOrg && ctxOrg) {
        const vctx = vpOriginal.getContext('2d');
        vpOriginal.width = cOrg.width;
        vpOriginal.height = cOrg.height;
        vctx.drawImage(cOrg, 0, 0);
      }

      const wrap = modal.querySelector('#vpAfterWrap');
      if (mime === 'application/pdf') {
        const ifr = document.createElement('iframe');
        ifr.src = url;
        ifr.style.cssText = 'width:100%; height:100%; background:white; border:0;';
        wrap.appendChild(ifr);
      } else if (mime && mime.includes('svg')) {
        const obj = document.createElement('object');
        obj.type = 'image/svg+xml';
        obj.data = url;
        obj.style.cssText = 'width:100%; height:100%;';
        wrap.appendChild(obj);
      } else if (mime && mime.startsWith('image/')) {
        const img = document.createElement('img');
        img.src = url;
        img.alt = filename || 'resultado';
        img.style.cssText = 'max-width:100%; height:auto; display:block;';
        wrap.appendChild(img);
      } else {
        const p = document.createElement('p');
        p.textContent = 'No se pudo previsualizar el resultado.';
        p.style.opacity = '0.8';
        wrap.appendChild(p);
      }

      const closeBtn = modal.querySelector('#vpClose');
      const cancelBtn= modal.querySelector('#vpCancel');
      const dl = modal.querySelector('#vpDownload');
      dl.href = (url || '').replace('/view/', '/download/');
      dl.download = filename || 'resultado';

      function cleanup(){ overlay.remove(); resolve(); }
      closeBtn?.addEventListener('click', cleanup);
      cancelBtn?.addEventListener('click', cleanup);
      overlay.addEventListener('click', e => { if (e.target === overlay) cleanup(); });
    });
  }

  // ====== Métricas de detalle (Mejorar) ======
  function computeDetailLevels(){
    try {
      if (!cRes || !ctxRes || !detailSpark) return;
      const w = cRes.width, h = cRes.height;
      if (!w || !h) return;
      const data = ctxRes.getImageData(0, 0, w, h).data;
      const gray = new Float32Array(w*h);
      for (let i=0, j=0; i<data.length; i+=4, j++){
        const r = data[i], g = data[i+1], b = data[i+2];
        gray[j] = (0.2126*r + 0.7152*g + 0.0722*b) / 255;
      }
      function blur(src, dst, k){
        const tmp = new Float32Array(w*h);
        for (let y=0;y<h;y++){
          let acc=0, cnt=0;
          for (let x=-k;x<=k;x++){ const xx=Math.min(w-1,Math.max(0,x)); acc+=src[y*w+xx]; cnt++; }
          tmp[y*w+0]=acc/cnt;
          for (let x=1;x<w;x++){
            const add = src[y*w+Math.min(w-1,x+k)];
            const rem = src[y*w+Math.max(0,x-1-k)];
            acc += add - rem;
            tmp[y*w+x] = acc/cnt;
          }
        }
        for (let x=0;x<w;x++){
          let acc=0, cnt=0;
          for (let y=-k;y<=k;y++){ const yy=Math.min(h-1,Math.max(0,y)); acc+=tmp[yy*w+x]; cnt++; }
          dst[0*w+x]=acc/cnt;
          for (let y=1;y<h;y++){
            const add = tmp[Math.min(h-1,y+k)*w+x];
            const rem = tmp[Math.max(0,y-1-k)*w+x];
            acc += add - rem;
            dst[y*w+x] = acc/cnt;
          }
        }
      }
      const g1 = new Float32Array(w*h), g2 = new Float32Array(w*h), g3 = new Float32Array(w*h);
      blur(gray, g1, 2); blur(gray, g2, 4); blur(gray, g3, 8);
      let eH=0, eM=0, eL=0;
      for (let i=0;i<w*h;i++){
        const hpf = gray[i]-g1[i];
        const mf  = g1[i]-g2[i];
        const lf  = g2[i]-g3[i];
        eH += hpf*hpf; eM += mf*mf; eL += lf*lf;
      }
      const tot = Math.max(1e-9, eH+eM+eL);
      const pH = Math.round(100*eH/tot);
      const pM = Math.round(100*eM/tot);
      const pL = Math.max(0, 100 - pH - pM);
      if (barHigh){ barHigh.style.width = pH + '%';  valHigh.textContent = pH + '%'; }
      if (barMid ){ barMid .style.width = pM + '%';  valMid .textContent = pM + '%'; }
      if (barLow ){ barLow .style.width = pL + '%';  valLow .textContent = pL + '%'; }

      function at(arr, x, y){ x=Math.max(0,Math.min(w-1,x)); y=Math.max(0,Math.min(h-1,y)); return arr[y*w+x]; }
      let maxMag=0; const mag = new Float32Array(w*h);
      for (let y=0;y<h;y++){
        for (let x=0;x<w;x++){
          const gx = -at(gray,x-1,y-1) -2*at(gray,x-1,y) -at(gray,x-1,y+1)
                     +at(gray,x+1,y-1) +2*at(gray,x+1,y) +at(gray,x+1,y+1);
          const gy = -at(gray,x-1,y-1) -2*at(gray,x,y-1) -at(gray,x+1,y-1)
                     +at(gray,x-1,y+1) +2*at(gray,x,y+1) +at(gray,x+1,y+1);
          const m = Math.hypot(gx, gy);
          mag[y*w+x] = m; if (m>maxMag) maxMag=m;
        }
      }
      const th = 0.25*maxMag;
      let edges=0; for (let i=0;i<w*h;i++) if (mag[i] >= th) edges++;
      const edgePct = Math.round(100 * edges / (w*h));
      if (barEdge){ barEdge.style.width = edgePct + '%'; valEdge.textContent = edgePct + '%'; }

      const s = detailSpark.getContext('2d');
      s.clearRect(0,0,detailSpark.width,detailSpark.height);
      const cols = detailSpark.width, colW = Math.max(1, Math.floor(w/cols));
      const line = new Float32Array(cols);
      for (let c=0;c<cols;c++){
        const x0 = c*colW, x1 = Math.min(w, x0+colW);
        let acc=0, n=0;
        for (let x=x0;x<x1;x++){
          for (let y=0;y<h;y++){
            const hpf = gray[y*w+x]-g1[y*w+x];
            acc += hpf*hpf; n++;
          }
        }
        line[c] = n? acc/n : 0;
      }
      const maxL = Math.max(...line) || 1e-6;
      s.fillStyle = '#0b1220'; s.fillRect(0,0,detailSpark.width,detailSpark.height);
      s.strokeStyle = 'rgba(255,255,255,0.06)'; s.beginPath();
      s.moveTo(0, detailSpark.height/2); s.lineTo(detailSpark.width, detailSpark.height/2); s.stroke();
      s.strokeStyle = '#22d3ee'; s.lineWidth = 2; s.beginPath();
      for (let c=0;c<cols;c++){
        const y = detailSpark.height - (line[c]/maxL)*(detailSpark.height-6) - 3;
        c===0 ? s.moveTo(0,y) : s.lineTo(c,y);
      }
      s.stroke();
    } catch (err) { console.error('[detalle] Error métricas:', err); }
  }
});
