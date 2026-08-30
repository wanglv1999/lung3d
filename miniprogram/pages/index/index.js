const { BASE } = require('../../config')

Page({
  data: {
    loading: false,
    error: '',
    cases: [],
  },

  onShow() {
    this.loadCases()
  },

  loadCases() {
    wx.request({
      url: BASE + '/api/cases',
      success: (res) => {
        if (res.statusCode === 200 && Array.isArray(res.data)) {
          this.setData({
            cases: res.data.map((c) => ({
              id: c.case_id,
              name: c.name,
              n: c.n,
              created: formatTime(c.created),
            })),
          })
        }
      },
      fail: () => {},
    })
  },

  chooseFile() {
    if (this.data.loading) return
    wx.chooseMessageFile({
      count: 1,
      type: 'file',
      extension: ['nii', 'gz', 'zip'],
      success: (res) => {
        const f = res.tempFiles && res.tempFiles[0]
        if (f) this.upload(f)
      },
      fail: () => {},
    })
  },

  upload(f) {
    this.setData({ loading: true, error: '' })
    wx.showLoading({ title: '上传处理中…', mask: true })
    wx.uploadFile({
      url: BASE + '/api/process',
      filePath: f.path,
      name: 'files',
      success: (res) => {
        wx.hideLoading()
        let info = null
        try { info = JSON.parse(res.data) } catch (e) { info = null }
        if (res.statusCode === 200 && info && info.structures) {
          wx.setStorageSync('caseInfo', info)
          wx.navigateTo({ url: '/pages/viewer/viewer' })
        } else {
          this.setData({
            loading: false,
            error: (info && info.error) || ('上传失败 HTTP ' + res.statusCode),
          })
        }
      },
      fail: (err) => {
        wx.hideLoading()
        this.setData({ loading: false, error: '上传失败：' + (err.errMsg || '') })
      },
    })
  },

  openCase(e) {
    const id = e.currentTarget.dataset.id
    wx.request({
      url: BASE + '/api/case/' + id,
      success: (res) => {
        if (res.statusCode === 200 && res.data && res.data.structures) {
          wx.setStorageSync('caseInfo', res.data)
          wx.navigateTo({ url: '/pages/viewer/viewer' })
        } else {
          wx.showToast({ title: '加载失败', icon: 'none' })
        }
      },
    })
  },
})

function formatTime(ts) {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  const p = (n) => (n < 10 ? '0' + n : '' + n)
  return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate()) + ' ' + p(d.getHours()) + ':' + p(d.getMinutes())
}