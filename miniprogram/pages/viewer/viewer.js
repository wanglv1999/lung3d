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
      // 扫码直达：按 scene 中的 case_id 从服务器拉取病例
      wx.showLoading({ title: '加载病例…', mask: true })
      wx.request({
        url: BASE + '/api/case/' + cid,
        success: (res) => {
          wx.hideLoading()
          if (res.statusCode === 200 && res.data && res.data.structures) {
            wx.setStorageSync('caseInfo', res.data)
            this.start()
          } else {
            wx.showToast({ title: '病例不存在或网络错误', icon: 'none' })
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
      wx.showToast({ title: '没有病例数据', icon: 'none' })
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
    const sys = wx.getSystemInfoSync()
    canvas.width = sys.windowWidth * sys.pixelRatio
    canvas.height = sys.windowHeight * sys.pixelRatio
    this.canvas = canvas

    const THREE = createScopedThreejs(canvas)
    this.THREE = THREE
    registerGLTFLoader(THREE)
    const { OrbitControls } = registerOrbit(THREE)

    const camera = new THREE.PerspectiveCamera(45, canvas.width / canvas.height, 0.1, 100000)
    camera.position.set(120, 120, 180)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0d1117)
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.1))
    const dir = new THREE.DirectionalLight(0xffffff, 1.2)
    dir.position.set(200, 300, 200)
    scene.add(dir)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(sys.pixelRatio)
    renderer.setSize(canvas.width, canvas.height)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08

    this.renderer = renderer
    this.scene = scene
    this.camera = camera
    this.controls = controls
    this.meshes = {}

    this.loadAll()

    const animate = () => {
      canvas.requestAnimationFrame(animate)
      controls.update()
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

    if (this.info.ct_mesh) {
      pending++
      loader.load(BASE + this.info.ct_mesh, (gltf) => {
        const g = gltf.scene
        g.traverse((o) => {
          if (o.isMesh) {
            o.material = new THREE.MeshStandardMaterial({
              color: 0x8892a0,
              transparent: true,
              opacity: 0.15,
              side: THREE.DoubleSide,
              depthWrite: false,
            })
          }
        })
        this.scene.add(g)
        this.ctGroup = g
        done()
      }, undefined, () => done())
    }

    for (const st of this.info.structures) {
      pending++
      const color = new THREE.Color(st.color[0], st.color[1], st.color[2])
      loader.load(BASE + st.mesh, (gltf) => {
        const g = gltf.scene
        g.traverse((o) => {
          if (o.isMesh) {
            o.material = new THREE.MeshStandardMaterial({
              color: color,
              roughness: 0.5,
              metalness: 0.05,
              transparent: true,
              opacity: 0.92,
              side: THREE.DoubleSide,
            })
          }
        })
        this.scene.add(g)
        this.meshes[st.key] = g
        done()
      }, undefined, () => done())
    }
  },

  fitCamera() {
    const THREE = this.THREE
    const box = new THREE.Box3()
    this.scene.traverse((o) => {
      if (o.isMesh && o.visible) {
        const b = new THREE.Box3().setFromObject(o)
        if (!b.isEmpty()) box.union(b)
      }
    })
    if (box.isEmpty()) return
    const size = new THREE.Vector3()
    box.getSize(size)
    const center = new THREE.Vector3()
    box.getCenter(center)
    const maxDim = Math.max(size.x, size.y, size.z) || 1
    const dist = maxDim * 1.9
    this.camera.position.set(center.x + dist * 0.7, center.y + dist * 0.6, center.z + dist)
    this.camera.near = Math.max(0.1, maxDim / 10000)
    this.camera.far = maxDim * 50
    this.camera.updateProjectionMatrix()
    this.controls.target.copy(center)
    this.controls.update()
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

  touchStart(e) {
    this.canvas.dispatchTouchEvent({ ...e, type: 'touchstart' })
  },
  touchMove(e) {
    this.canvas.dispatchTouchEvent({ ...e, type: 'touchmove' })
  },
  touchEnd(e) {
    this.canvas.dispatchTouchEvent({ ...e, type: 'touchend' })
  },
})