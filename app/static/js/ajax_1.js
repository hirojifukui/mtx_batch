function analyze() {
  // console.log("Hello from Ajax-1.js!");
  $.ajax({
    type: 'POST',
    url: '/analyze',
    data: '',
    contentType: 'application/json',
    success: function (data) {
      const eval = JSON.parse(data.ResultSet).eval
      const img = JSON.parse(data.ResultSet).img
      document.getElementById('all').innerHTML = eval
    //  document.getElementById('greeting_image').src = greeting_image
    }
  })
}