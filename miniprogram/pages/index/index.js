const { BASE, ICP_BEIAN } = require('../../config')

Page({
  data: {
    loading: false,
    error: '',
    cases: [],
    icpBeian: ICP_BEIAN,
  },

  onShow() {
    this.loadCases()
  },

  copyBeian() {
    if (!this.data.icpBeian) return
    wx.setClipboardData({
      data: this.data.icpBeian,
      success: () => wx.showToast({ title: '备案号已复制', icon: 'none' }),
    })
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
    wx.showModal({
      title: '数据名称',
      content: '给该数据起个名字（可选）：',
      editable: true,
      placeholderText: '如：张三 2026-09-01',
      success: (res) => {
        const name = (res.confirm && res.content && res.content.trim()) || ''
        this.doUpload(f, name)
      },
      fail: () => {},
    })
  },

  doUpload(f, name) {
    this.setData({ loading: true, error: '' })
    wx.showLoading({ title: '上传处理中…', mask: true })
    wx.uploadFile({
      url: BASE + '/api/process',
      filePath: f.path,
      name: 'files',
      formData: { name: name || '', filename: f.name || '' },
      timeout: 300000,
      success: (res) => {
        let info = null
        try { info = JSON.parse(res.data) } catch (e) { info = null }
        if (res.statusCode === 200 && info && info.status === 'processing') {
          this.waitCase(info.case_id, 0)
        } else if (res.statusCode === 200 && info && info.structures) {
          wx.hideLoading()
          wx.setStorageSync('caseInfo', info)
          wx.navigateTo({ url: '/pages/viewer/viewer' })
        } else {
          wx.hideLoading()
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

  waitCase(caseId, tries) {
    wx.showLoading({ title: '处理中…', mask: true })
    wx.request({
      url: BASE + '/api/case/' + caseId,
      timeout: 20000,
      success: (res) => {
        if (res.statusCode === 200 && res.data && res.data.structures) {
          wx.hideLoading()
          wx.setStorageSync('caseInfo', res.data)
          wx.navigateTo({ url: '/pages/viewer/viewer' })
        } else if (res.statusCode === 400 && res.data && res.data.error) {
          wx.hideLoading()
          this.setData({ loading: false, error: res.data.error })
        } else if (tries > 250) {
          wx.hideLoading()
          this.setData({ loading: false, error: '处理超时，请稍后重试' })
        } else {
          setTimeout(() => this.waitCase(caseId, tries + 1), 2000)
        }
      },
      fail: () => {
        wx.hideLoading()
        this.setData({ loading: false, error: '网络错误' })
      },
    })
  },

  renameCase(e) {
    const id = e.currentTarget.dataset.id
    wx.showModal({
      title: '重命名',
      editable: true,
      placeholderText: '输入新的数据名称',
      success: (res) => {
        if (res.confirm && res.content && res.content.trim()) {
          wx.request({
            url: BASE + '/api/case/' + id + '/rename',
            method: 'POST',
            data: { name: res.content.trim() },
            header: { 'content-type': 'application/json' },
            success: (r) => {
              if (r.statusCode === 200) {
                this.loadCases()
                wx.showToast({ title: '已重命名', icon: 'success' })
              } else {
                wx.showToast({ title: '重命名失败', icon: 'none' })
              }
            },
            fail: () => wx.showToast({ title: '网络错误', icon: 'none' }),
          })
        }
      },
    })
  },

  deleteCase(e) {
    const id = e.currentTarget.dataset.id
    wx.showModal({
      title: '删除数据',
      content: '确定删除该数据？删除后不可恢复。',
      success: (res) => {
        if (res.confirm) {
          wx.request({
            url: BASE + '/api/case/' + id,
            method: 'DELETE',
            success: (r) => {
              if (r.statusCode === 200) {
                this.loadCases()
                wx.showToast({ title: '已删除', icon: 'success' })
              } else {
                wx.showToast({ title: '删除失败', icon: 'none' })
              }
            },
            fail: () => wx.showToast({ title: '删除失败', icon: 'none' }),
          })
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