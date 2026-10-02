import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const params = new URLSearchParams(location.search);
const BASE = params.get('api') || location.origin;
const CASE_ID = params.get('case');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 20000);
camera.position.set(120, 120, 180);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.getElementById('viewport').appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 1.1));
const dir = new THREE.DirectionalLight(0xffffff, 1.2);
dir.position.set(200, 300, 200);
scene.add(dir);
scene.add(new THREE.DirectionalLight(0xffffff, 0.4).translateX(-200).translateY(-150).translateZ(-150));

// 模型承载组：旋转 pivot 实现"相对视角固定"的旋转（与小程序的轨迹球一致）
const pivot = new THREE.Group();
scene.add(pivot);
const modelGroup = new THREE.Group();
pivot.add(modelGroup);
const camTarget = new THREE.Vector3(0, 0, 0);

const meshes = [];          // {key, group, visible}
let fitBox = null;
let currentCaseId = '';

const $ = (id) => document.getElementById(id);
const apiBase = BASE;

function showError(msg) {
  const e = $('error');
  e.hidden = false;
  e.textContent = msg;
}
function clearError() { $('error').hidden = true; }

function fitCamera() {
  const savedPos = modelGroup.position.clone();
  const savedQ = pivot.quaternion.clone();
  modelGroup.position.set(0, 0, 0);
  pivot.quaternion.identity();
  pivot.updateMatrixWorld(true);
  const box = new THREE.Box3();
  modelGroup.traverse((o) => {
    if (o.isMesh && o.visible) {
      const b = new THREE.Box3().setFromObject(o);
      if (!b.isEmpty()) box.union(b);
    }
  });
  modelGroup.position.copy(savedPos);
  pivot.quaternion.copy(savedQ);
  if (box.isEmpty()) return;
  const size = new THREE.Vector3();
  box.getSize(size);
  const center = new THREE.Vector3();
  box.getCenter(center);
  // 模型中心移到原点，相机围绕原点旋转
  modelGroup.position.set(-center.x, -center.y, -center.z);
  camTarget.set(0, 0, 0);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const dist = maxDim * 1.9;
  camera.position.set(dist * 0.7, dist * 0.6, dist);
  camera.near = Math.max(0.1, maxDim / 10000);
  camera.far = maxDim * 50;
  camera.updateProjectionMatrix();
  camera.lookAt(camTarget);
}

function fitChanged() {
  if (meshes.length === 0) return;
  const box = new THREE.Box3();
  for (const m of meshes) {
    if (m.group.visible) {
      const b = new THREE.Box3().setFromObject(m.group);
      if (!b.isEmpty()) box.union(b);
    }
  }
  if (!box.isEmpty()) { fitBox = box; }
}

function addStructure(st) {
  const group = new THREE.Group();
  modelGroup.add(group);
  const loader = new GLTFLoader();
  loader.load(apiBase + st.mesh, (gltf) => {
    gltf.scene.traverse((o) => {
      if (o.isMesh) {
        const mat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(st.color[0], st.color[1], st.color[2]),
          roughness: 0.45,
          metalness: 0.05,
          transparent: true,
          opacity: 0.92,
          side: THREE.DoubleSide,
        });
        o.material = mat;
        o.castShadow = true;
      }
    });
    group.add(gltf.scene);
    fitChanged();
    fitCamera();
  }, undefined, (err) => {
    showError('加载网格失败：' + st.name + ' ' + (err && err.message || ''));
  });
  const entry = { key: st.key, group, visible: true };
  meshes.push(entry);
  renderStructRow(entry, st);
}

function addCt(meshUrl) {
  const group = new THREE.Group();
  modelGroup.add(group);
  const loader = new GLTFLoader();
  loader.load(apiBase + meshUrl, (gltf) => {
    gltf.scene.traverse((o) => {
      if (o.isMesh) {
        o.material = new THREE.MeshStandardMaterial({
          color: new THREE.Color(0x8892a0),
          roughness: 0.8,
          metalness: 0.0,
          transparent: true,
          opacity: 0.18,
          side: THREE.DoubleSide,
          depthWrite: false,
        });
      }
    });
    group.add(gltf.scene);
    const entry = { key: '__ct__', group, visible: true };
    meshes.unshift(entry);
    fitChanged();
    fitCamera();
  }, undefined, () => showError('加载 CT 轮廓失败'));
}

function renderStructRow(entry, st) {
  const item = document.createElement('label');
  item.className = 'struct';
  const sw = document.createElement('span');
  sw.className = 'swatch';
  sw.style.background = `rgb(${st.color[0]*255|0},${st.color[1]*255|0},${st.color[2]*255|0})`;
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = true;
  cb.addEventListener('change', () => {
    entry.visible = cb.checked;
    entry.group.visible = cb.checked;
    fitChanged();
  });
  const name = document.createElement('span');
  name.className = 'sname';
  name.textContent = st.name;
  const vol = document.createElement('span');
  vol.className = 'svol';
  vol.textContent = (st.volume_cm3 ?? 0).toFixed(1) + ' cm³';
  item.append(sw, cb, name, vol);
  $('struct-list').appendChild(item);
}

function renderCase(info) {
  $('case-panel').hidden = false;
  $('case-name').textContent = info.name || info.case_id;
  currentCaseId = info.case_id || '';
  const meta = [];
  if (info.dimensions) meta.push(info.dimensions.join('×'));
  if (info.spacing_mm) meta.push('体素 ' + info.spacing_mm.map(v => v.toFixed(1)).join('×') + ' mm');
  $('case-meta').textContent = meta.join(' ｜ ');

  $('struct-list').innerHTML = '';
  meshes.length = 0;
  fitBox = null;
  pivot.quaternion.identity();
  modelGroup.position.set(0, 0, 0);
  while (modelGroup.children.length) modelGroup.remove(modelGroup.children[0]);

  if (info.ct_mesh) addCt(info.ct_mesh);
  for (const st of info.structures) addStructure(st);
  if (info.structures.length === 0) showError('未提取到结构');
}

function uploadFiles(files) {
  if (!files || files.length === 0) return;
  clearError();
  $('progress').hidden = false;
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  fetch(apiBase + '/api/process', { method: 'POST', body: fd })
    .then(async (r) => {
      const data = await r.json().catch(() => null);
      if (!r.ok) throw new Error((data && data.error) || ('HTTP ' + r.status));
      return data;
    })
    .then((info) => {
      if (info && info.status === 'processing') {
        waitCase(info.case_id, 0);
      } else {
        $('progress').hidden = true;
        renderCase(info);
      }
    })
    .catch((err) => {
      $('progress').hidden = true;
      showError('处理失败：' + err.message);
    });
}

let waitTimer = null;
function waitCase(caseId, tries) {
  $('progress').textContent = '处理中…请稍候';
  fetch(apiBase + '/api/case/' + encodeURIComponent(caseId))
    .then(async (r) => {
      const data = await r.json().catch(() => null);
      if (r.status === 200 && data && data.structures) {
        $('progress').hidden = true;
        renderCase(data);
      } else if (r.status === 400 && data && data.error) {
        $('progress').hidden = true;
        showError(data.error);
      } else if (tries > 90) {
        $('progress').hidden = true;
        showError('处理超时，请稍后重试');
      } else {
        waitTimer = setTimeout(() => waitCase(caseId, tries + 1), 2000);
      }
    })
    .catch(() => {
      $('progress').hidden = true;
      showError('网络错误');
    });
}

// ---------- 自定义鼠标控制：模型绕屏幕轴旋转（与小程序一致）----------
const SPEED = 0.005;
let drag = null;

function orbit(dx, dy) {
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
  const q = new THREE.Quaternion().setFromAxisAngle(up, dx * SPEED);
  q.multiply(new THREE.Quaternion().setFromAxisAngle(right, dy * SPEED));
  pivot.quaternion.premultiply(q);
}
function zoom(scale) {
  const d = new THREE.Vector3().subVectors(camTarget, camera.position).normalize();
  const dist = Math.max(camera.position.distanceTo(camTarget) * scale, 1);
  camera.position.copy(camTarget).addScaledVector(d, -dist);
  camera.lookAt(camTarget);
}
function pan(dx, dy) {
  const k = 0.002 * camera.position.distanceTo(camTarget);
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0).multiplyScalar(-dx * k);
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1).multiplyScalar(dy * k);
  const delta = right.add(up);
  camTarget.add(delta);
  camera.position.add(delta);
}

renderer.domElement.addEventListener('mousedown', (e) => {
  if (e.button === 0) drag = { mode: 'orbit', x: e.clientX, y: e.clientY };
  else if (e.button === 2) drag = { mode: 'pan', x: e.clientX, y: e.clientY };
});
window.addEventListener('mousemove', (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x;
  const dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.mode === 'orbit') orbit(dx, dy);
  else if (drag.mode === 'pan') pan(dx, dy);
});
window.addEventListener('mouseup', () => { drag = null; });
renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());
renderer.domElement.addEventListener('wheel', (e) => {
  e.preventDefault();
  zoom(e.deltaY > 0 ? 1.1 : 1 / 1.1);
}, { passive: false });
let touchState = null;

renderer.domElement.addEventListener('touchstart', (e) => {
  const ts = e.touches;
  if (ts.length >= 2) {
    touchState = {
      x1: ts[0].clientX, y1: ts[0].clientY,
      x2: ts[1].clientX, y2: ts[1].clientY,
      dist: Math.hypot(ts[0].clientX - ts[1].clientX, ts[0].clientY - ts[1].clientY),
    };
  } else if (ts.length === 1) {
    touchState = { x1: ts[0].clientX, y1: ts[0].clientY, dist: 0 };
  }
  e.preventDefault();
}, { passive: false });

renderer.domElement.addEventListener('touchmove', (e) => {
  if (!touchState) return;
  const ts = e.touches;
  if (ts.length >= 2) {
    const x2 = ts[1].clientX, y2 = ts[1].clientY;
    const d = Math.hypot(ts[0].clientX - x2, ts[0].clientY - y2);
    if (touchState.dist > 0) zoom(touchState.dist / Math.max(d, 1));
    const dx = ((ts[0].clientX - touchState.x1) + (x2 - touchState.x2)) / 2;
    const dy = ((ts[0].clientY - touchState.y1) + (y2 - touchState.y2)) / 2;
    pan(dx, dy);
    touchState.x1 = ts[0].clientX; touchState.y1 = ts[0].clientY;
    touchState.x2 = x2; touchState.y2 = y2; touchState.dist = d;
  } else if (ts.length === 1) {
    const dx = ts[0].clientX - touchState.x1;
    const dy = ts[0].clientY - touchState.y1;
    orbit(dx, dy);
    touchState.x1 = ts[0].clientX; touchState.y1 = ts[0].clientY;
  }
  e.preventDefault();
}, { passive: false });

renderer.domElement.addEventListener('touchend', () => { touchState = null; });

$('file-input').addEventListener('change', (e) => uploadFiles(e.target.files));
$('btn-upload-more').addEventListener('click', () => $('file-input').click());
$('upload-box').addEventListener('click', (e) => { if (e.target === $('upload-hint')) $('file-input').click(); });
$('btn-reset').addEventListener('click', fitCamera);
$('btn-all').addEventListener('click', () => { meshes.forEach(m => { m.visible = true; m.group.visible = true; }); document.querySelectorAll('#struct-list input[type=checkbox]').forEach(c => c.checked = true); });
$('btn-none').addEventListener('click', () => { meshes.forEach(m => { m.visible = false; m.group.visible = false; }); document.querySelectorAll('#struct-list input[type=checkbox]').forEach(c => c.checked = false); });

async function showQr(url) {
  $('qr-modal').classList.add('open');
  $('qr-img').style.display = 'none';
  $('qr-hint').textContent = '生成中…';
  try {
    const r = await fetch(url);
    if (!r.ok) {
      const err = await r.json().catch(() => null);
      $('qr-hint').textContent = (err && err.error) || ('HTTP ' + r.status);
      return;
    }
    const blob = await r.blob();
    $('qr-img').src = URL.createObjectURL(blob);
    $('qr-img').style.display = 'block';
    $('qr-hint').textContent = '用微信“扫一扫”打开';
  } catch (e) {
    $('qr-hint').textContent = '生成失败：' + e.message;
  }
}

$('btn-wxcode').addEventListener('click', () => {
  if (!currentCaseId) { showError('请先上传或打开一个病例'); return; }
  showQr(apiBase + '/api/wxacode?case=' + encodeURIComponent(currentCaseId));
});
$('btn-qr-wx').addEventListener('click', () => {
  if (!currentCaseId) return;
  showQr(apiBase + '/api/wxacode?case=' + encodeURIComponent(currentCaseId));
});
$('btn-qr-url').addEventListener('click', () => {
  if (!currentCaseId) return;
  const base = location.origin;
  showQr(apiBase + '/api/qrcode?case=' + encodeURIComponent(currentCaseId) + '&base=' + encodeURIComponent(base));
});
$('qr-close').addEventListener('click', () => { $('qr-modal').classList.remove('open'); });

function resize() {
  const v = $('viewport');
  const w = v.clientWidth, h = v.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

(function animate() {
  requestAnimationFrame(animate);
  camera.lookAt(camTarget);
  renderer.render(scene, camera);
})();

if (CASE_ID) {
  fetch(apiBase + '/api/case/' + encodeURIComponent(CASE_ID))
    .then(r => r.json())
    .then(info => { if (info && info.structures) renderCase(info); else showError('未找到该病例'); })
    .catch(() => showError('加载病例失败'));
}
