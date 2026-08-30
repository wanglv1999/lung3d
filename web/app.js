import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
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

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.AmbientLight(0xffffff, 1.1));
const dir = new THREE.DirectionalLight(0xffffff, 1.2);
dir.position.set(200, 300, 200);
scene.add(dir);
scene.add(new THREE.DirectionalLight(0xffffff, 0.4).translateX(-200).translateY(-150).translateZ(-150));

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
  if (!fitBox) return;
  const box = fitBox.clone();
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const dist = maxDim * 1.9;
  camera.position.set(center.x + dist * 0.7, center.y + dist * 0.6, center.z + dist);
  camera.near = Math.max(0.1, maxDim / 10000);
  camera.far = maxDim * 50;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
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
  scene.add(group);
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
  scene.add(group);
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
    fitCamera();
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
  scene.clear();
  scene.add(new THREE.AmbientLight(0xffffff, 1.1));
  const d1 = new THREE.DirectionalLight(0xffffff, 1.2);
  d1.position.set(200, 300, 200);
  scene.add(d1);
  scene.add(new THREE.DirectionalLight(0xffffff, 0.4).translateX(-200).translateY(-150).translateZ(-150));

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
      $('progress').hidden = true;
      renderCase(info);
    })
    .catch((err) => {
      $('progress').hidden = true;
      showError('处理失败：' + err.message);
    });
}

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
    $('qr-hint').textContent = '用微信"扫一扫"打开';
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
  controls.update();
  renderer.render(scene, camera);
})();

if (CASE_ID) {
  fetch(apiBase + '/api/case/' + encodeURIComponent(CASE_ID))
    .then(r => r.json())
    .then(info => { if (info && info.structures) renderCase(info); else showError('未找到该病例'); })
    .catch(() => showError('加载病例失败'));
}