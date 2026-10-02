const { createScopedThreejs } = require('../../libs/threejs-miniprogram.js')
const registerGLTFLoader = require('../../libs/gltf-loader')
const registerOrbit = require('../../libs/orbit')
const { BASE } = require('../../config')

function resolveCaseId(options) {
  let cid = options.c
  if (!cid && options.scene) {
    let s = options.scene
    try { s = decodeURIComponent(s) } catch (e) {}
    const m = s.match(/c=([A-Za-z0-9_]+)/)
    if (m) cid = m[1]
  }
  return cid || ''
}

Page({
  data: {
    name: '',
    meta: '',
    loading: true,
    structures: [],
  },

  onLoad(options) {
    options = options || {}
    const cid = resolveCaseId(options)
    if (cid) {
      // 扫码直达：按 scene 中的 case_id 从服务器拉取数据
      wx.showLoading({ title: '加载数据…', mask: true })
      wx.request({
        url: BASE + '/api/case/' + cid,
        success: (res) => {
          wx.hideLoading()
          if (res.statusCode === 200 && res.data && res.data.structures) {
            wx.setStorageSync('caseInfo', res.data)
            this.start()
          } else {
            wx.showToast({ title: '数据不存在或网络错误', icon: 'none' })
            setTimeout(() => wx.navigateBack(), 1200)
          }
        },
        fail: () => {
          wx.hideLoading()
          wx.showToast({ title: '网络错误', icon: 'none' })
          setTimeout(() => wx.navigateBack(), 1200)
        },
      })
    } else {
      this.start()
    }
  },

  start() {
    this.info = wx.getStorageSync('caseInfo') || null
    if (!this.info || !this.info.structures) {
      wx.showToast({ title: '没有数据', icon: 'none' })
      wx.navigateBack()
      return
    }
    const meta = []
    if (this.info.dimensions) meta.push(this.info.dimensions.join('×'))
    if (this.info.spacing_mm) meta.push('体素 ' + this.info.spacing_mm.map((v) => v.toFixed(1)).join('×') + 'mm')
    const structures = []
    if (this.info.ct_mesh) {
      structures.push({ key: '__ct__', name: 'CT轮廓', color: 'rgb(136,146,160)', checked: true })
    }
    for (const s of this.info.structures) {
      structures.push({
        key: s.key,
        name: s.name,
        color: 'rgb(' + s.color.map((c) => Math.round(c * 255)).join(',') + ')',
        checked: true,
      })
    }
    this.setData({ name: this.info.name || this.info.case_id, meta: meta.join(' '), structures })
  },

  onReady() {
    wx.createSelectorQuery().select('#webgl').node().exec((res) => {
      if (res && res[0] && res[0].node) this.init3d(res[0].node)
    })
  },

  init3d(canvas) {
    const win = wx.getWindowInfo()
    const dpr = Math.min(win.pixelRatio || 2, 2)
    canvas.width = Math.min(win.windowWidth, 16384)
    canvas.height = Math.min(win.windowHeight, 16384)
    this.canvas = canvas

    const THREE = createScopedThreejs(canvas)
    this.THREE = THREE
    registerGLTFLoader(THREE)

    const camera = new THREE.PerspectiveCamera(45, canvas.width / canvas.height, 0.1, 100000)
    camera.position.set(120, 120, 180)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0d1117)
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.1))
    const dir = new THREE.DirectionalLight(0xffffff, 1.2)
    dir.position.set(200, 300, 200)
    scene.add(dir)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(dpr)
    renderer.setSize(canvas.width, canvas.height)

    this.renderer = renderer
    this.scene = scene
    this.camera = camera
    this.camTarget = new THREE.Vector3(0, 0, 0)
    this.meshes = {}
    const pivot = new THREE.Group()
    scene.add(pivot)
    this.pivot = pivot
    const modelGroup = new THREE.Group()
    pivot.add(modelGroup)
    this.modelGroup = modelGroup

    this.loadAll()

    const animate = () => {
      canvas.requestAnimationFrame(animate)
      camera.lookAt(this.camTarget)
      renderer.render(scene, camera)
    }
    animate()
  },

  loadAll() {
    const THREE = this.THREE
    const loader = new THREE.GLTFLoader()
    let pending = 0
    const done = () => {
      pending--
      if (pending <= 0) {
        this.fitCamera()
        this.setData({ loading: false })
      }
    }

    const addMesh = (url, color, opacity, depthWrite, key) => {
      pending++
      wx.downloadFile({
        url: BASE + url,
        success: (res) => {
          if (res.statusCode !== 200) {
            console.error('download fail', url, res.statusCode)
            done()
            return
          }
          wx.getFileSystemManager().readFile({
            filePath: res.tempFilePath,
            success: (r) => {
              loader.parse(r.data, '', (gltf) => {
                const g = gltf.scene
                g.traverse((o) => {
                  if (o.isMesh) {
                    o.material = new THREE.MeshStandardMaterial({
                      color: color,
                      roughness: 0.5,
                      metalness: 0.05,
                      transparent: true,
                      opacity: opacity,
                      side: THREE.DoubleSide,
                      depthWrite: depthWrite,
                    })
                  }
                })
                this.modelGroup.add(g)
                if (key === '__ct__') this.ctGroup = g
                else this.meshes[key] = g
                done()
              }, (e) => {
                console.error('parse fail', url, e)
                done()
              })
            },
            fail: (e) => { console.error('readFile fail', url, e); done() },
          })
        },
        fail: (e) => { console.error('downloadFile fail', url, e); done() },
      })
    }

    if (this.info.ct_mesh) {
      addMesh(this.info.ct_mesh, 0x8892a0, 0.15, false, '__ct__')
    }
    for (const st of this.info.structures) {
      const color = new THREE.Color(st.color[0], st.color[1], st.color[2])
      addMesh(st.mesh, color, 0.92, true, st.key)
    }
  },
  fitCamera() {
    const THREE = this.THREE
    const savedPos = this.modelGroup.position.clone()
    const savedQ = this.pivot.quaternion.clone()
    this.modelGroup.position.set(0, 0, 0)
    this.pivot.quaternion.set(0, 0, 0, 1)
    this.pivot.updateMatrixWorld(true)
    const box = new THREE.Box3()
    this.modelGroup.traverse((o) => {
      if (o.isMesh && o.visible) {
        const b = new THREE.Box3().setFromObject(o)
        if (!b.isEmpty()) box.union(b)
      }
    })
    this.modelGroup.position.copy(savedPos)
    this.pivot.quaternion.copy(savedQ)
    if (box.isEmpty()) return
    const size = new THREE.Vector3()
    box.getSize(size)
    const center = new THREE.Vector3()
    box.getCenter(center)
    this.modelGroup.position.set(-center.x, -center.y, -center.z)
    this.camTarget.set(0, 0, 0)
    const maxDim = Math.max(size.x, size.y, size.z) || 1
    const dist = maxDim * 1.9
    this.camera.position.set(dist * 0.7, dist * 0.6, dist)
    this.camera.near = Math.max(0.1, maxDim / 10000)
    this.camera.far = maxDim * 50
    this.camera.updateProjectionMatrix()
    this.camera.lookAt(this.camTarget)
  },

  setVisible(key, visible) {
    if (key === '__ct__') {
      if (this.ctGroup) this.ctGroup.visible = visible
    } else if (this.meshes[key]) {
      this.meshes[key].visible = visible
    }
  },

  onToggle(e) {
    const key = e.currentTarget.dataset.key
    const idx = this.data.structures.findIndex((s) => s.key === key)
    if (idx < 0) return
    const checked = !this.data.structures[idx].checked
    this.setData({ ['structures[' + idx + '].checked']: checked })
    this.setVisible(key, checked)
  },

  onReset() {
    this.fitCamera()
  },

  onAll() {
    const structures = this.data.structures.map((s) => ({ ...s, checked: true }))
    this.setData({ structures })
    for (const s of structures) this.setVisible(s.key, true)
  },

  onNone() {
    const structures = this.data.structures.map((s) => ({ ...s, checked: false }))
    this.setData({ structures })
    for (const s of structures) this.setVisible(s.key, false)
  },

  goBack() {
    const pages = getCurrentPages()
    if (pages.length > 1) {
      wx.navigateBack()
    } else {
      wx.reLaunch({ url: '/pages/index/index' })
    }
  },

  touchStart(e) {
    const ts = e.touches || []
    this._touch = { x1: ts[0] && ts[0].pageX, y1: ts[0] && ts[0].pageY, dist: 0 }
    if (ts.length >= 2) {
      this._touch.x2 = ts[1].pageX
      this._touch.y2 = ts[1].pageY
      this._touch.dist = Math.hypot(ts[0].pageX - ts[1].pageX, ts[0].pageY - ts[1].pageY)
    }
  },
  touchMove(e) {
    const ts = e.touches || []
    const t = this._touch
    if (!t || ts.length === 0) return
    if (ts.length >= 2) {
      const x2 = ts[1].pageX
      const y2 = ts[1].pageY
      const d = Math.hypot(ts[0].pageX - x2, ts[0].pageY - y2)
      if (t.dist > 0) this.zoom(t.dist / Math.max(d, 1))
      const dx = ((ts[0].pageX - t.x1) + (x2 - t.x2)) / 2
      const dy = ((ts[0].pageY - t.y1) + (y2 - t.y2)) / 2
      this.pan(dx, dy)
      t.x1 = ts[0].pageX; t.y1 = ts[0].pageY; t.x2 = x2; t.y2 = y2; t.dist = d
    } else {
      const dx = ts[0].pageX - t.x1
      const dy = ts[0].pageY - t.y1
      this.orbit(dx, dy)
      t.x1 = ts[0].pageX; t.y1 = ts[0].pageY
    }
  },
  touchEnd(e) {
    this._touch = null
  },

  orbit(dx, dy) {
    const THREE = this.THREE
    const SPEED = 0.005
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0)
    const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1)
    const q = new THREE.Quaternion().setFromAxisAngle(up, dx * SPEED)
    q.multiply(new THREE.Quaternion().setFromAxisAngle(right, dy * SPEED))
    this.pivot.quaternion.premultiply(q)
  },
  zoom(scale) {
    const THREE = this.THREE
    const dir = new THREE.Vector3().subVectors(this.camTarget, this.camera.position).normalize()
    const dist = Math.max(this.camera.position.distanceTo(this.camTarget) * scale, 1)
    this.camera.position.copy(this.camTarget).addScaledVector(dir, -dist)
    this.camera.lookAt(this.camTarget)
  },
  pan(dx, dy) {
    const THREE = this.THREE
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0).multiplyScalar(-dx * 0.5)
    const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1).multiplyScalar(dy * 0.5)
    const delta = right.add(up)
    this.camTarget.add(delta)
    this.camera.position.add(delta)
  },
})
