// courtesy of ahmed-musallam  'https://gist.github.com/ahmed-musallam/d0378d3494744d412cb7b69a3313e2da'
var consoleElem = document.querySelector('#console')
var parentElem = consoleElem.parentElement;
function log(txt,toEnd=false) {
	var newLine = document.createElement('li');
	newLine.innerHTML = typeof txt === 'string' ? txt : JSON.stringify(txt, null, 4);
	consoleElem.appendChild(newLine);
	if(toEnd===true)
		parentElem.scrollTop = parentElem.scrollHeight;
}
