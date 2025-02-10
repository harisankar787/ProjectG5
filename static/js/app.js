document.getElementById('videoForm').addEventListener('submit', function(e) {
  e.preventDefault();
  var videoId = document.getElementById('video_id').value;

  fetch('/analyze', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: 'video_id=' + videoId
  })
  .then(response => response.json())
  .then(data => {
      if (data.error) {
          document.getElementById('error').style.display = 'block';
          document.getElementById('results').style.display = 'none';
      } else {
          document.getElementById('positive').innerText = data.positive;
          document.getElementById('neutral').innerText = data.neutral;
          document.getElementById('negative').innerText = data.negative;
          document.getElementById('results').style.display = 'block';
          document.getElementById('error').style.display = 'none';
      }
  })
  .catch(error => {
      console.error(error);
      document.getElementById('error').style.display = 'block';
      document.getElementById('results').style.display = 'none';
  });
});
