let {PythonShell} = require('python-shell')
var path = require("path")

function run_local() {

  var alert_mode = document.getElementById("alert_checkbox").checked
  document.getElementById("buton_ex_locala").innerHTML = '<span class="spinner-grow spinner-grow-sm" id = "loading_circle_local" role="status" aria-hidden="true" ></span> In executie...'
  console.log('The alert mode was :  ' + alert_mode);
  var options = {
    scriptPath : path.join(__dirname, '/../back_end/'),
    pythonPath: 'C:\\Users\\georg\\AppData\\Local\\Microsoft\\WindowsApps\\python3.exe',
    args:[alert_mode]
  }

  let pyshell = new PythonShell('local_detection.py', options);
  pyshell.end(function (err,code,signal) {
  document.getElementById("buton_ex_locala").innerHTML = 'Executie locala'
  if (err) throw err;
  console.log('The exit code was: ' + code);
  console.log('The exit signal was: ' + signal);
  console.log('finished local detection');
});

}

function run_in_cloud() {

  var alert_mode = document.getElementById("alert_checkbox").checked
  document.getElementById("buton_ex_cloud").innerHTML = '<span class="spinner-grow spinner-grow-sm" id = "loading_circle_local" role="status" aria-hidden="true" ></span> In executie...'

  var options = {
    scriptPath : path.join(__dirname, '/../back_end/'),
    pythonPath: 'C:\\Users\\georg\\AppData\\Local\\Microsoft\\WindowsApps\\python3.exe',
    args : [alert_mode]
  }

  let pyshell = new PythonShell('cloud_detection.py', options);
   pyshell.end(function (err,code,signal) {
   document.getElementById("buton_ex_cloud").innerHTML = "Executie in cloud"
  if (err) throw err;
  console.log('The exit code was: ' + code);
  console.log('The exit signal was: ' + signal);
  console.log('finished cloud detection');
});
}