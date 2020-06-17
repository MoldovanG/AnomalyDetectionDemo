let {PythonShell} = require('python-shell')
var path = require("path")

function run_local() {

  var alert_mode = document.getElementById("alert_checkbox").value
  console.log('The alert mode was :  ' + alert_mode);
  var options = {
    scriptPath : path.join(__dirname, '/../back_end/'),
    pythonPath: 'C:\\Users\\georg\\AppData\\Local\\Microsoft\\WindowsApps\\python3.exe',
    args:[alert_mode]
  }

  let pyshell = new PythonShell('local_detection.py', options);

  pyshell.end(function (err,code,signal) {
  if (err) throw err;
  console.log('The exit code was: ' + code);
  console.log('The exit signal was: ' + signal);
  console.log('finished local detection');
});

}

function run_in_cloud() {

  var alert_mode = document.getElementById("alert_checkbox").value

  var options = {
    scriptPath : path.join(__dirname, '/../back_end/'),
    pythonPath: 'C:\\Users\\georg\\AppData\\Local\\Microsoft\\WindowsApps\\python3.exe',
    args : [alert_mode]
  }

  let pyshell = new PythonShell('cloud_detection.py', options);
   pyshell.end(function (err,code,signal) {
  if (err) throw err;
  console.log('The exit code was: ' + code);
  console.log('The exit signal was: ' + signal);
  console.log('finished cloud detection');
});
}