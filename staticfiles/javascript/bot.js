$(document).ready(function () {
  var trigger = $('.hamburger'),
      overlay = $('.overlay'),
     isClosed = false;

    trigger.click(function () {
      hamburger_cross();      
    });

    function hamburger_cross() {

      if (isClosed == true) {          
        overlay.hide();
        trigger.removeClass('is-open');
        trigger.addClass('is-closed');
        isClosed = false;
      } else {   
        overlay.show();
        trigger.removeClass('is-closed');
        trigger.addClass('is-open');
        isClosed = true;
      }
  }
  
  $('[data-toggle="offcanvas"]').click(function () {
        $('#wrapper').toggleClass('toggled');
  });  

  setColorByMood(mood)

});


function getCookie(name) {
  var cookieValue = null;
  if (document.cookie && document.cookie != '') {
      var cookies = document.cookie.split(';');
      for (var i = 0; i < cookies.length; i++) {
          var cookie = jQuery.trim(cookies[i]);
          // Does this cookie string begin with the name we want?
          if (cookie.substring(0, name.length + 1) == (name + '=')) {
              cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
              break;
          }
      }
  }
  return cookieValue;
}

$.ajaxSetup({
  beforeSend: function(xhr, settings) {
      if (!(/^http:.*/.test(settings.url) || /^https:.*/.test(settings.url))) { 
          // Only send the token to relative URLs i.e. locally.
         xhr.setRequestHeader("X-CSRFToken", getCookie('csrftoken'));
      }
  }
});


function sendMessage() {
  if  ($('#messageToBot').val() != '') {
    $.ajax({
      type: 'POST',
      url: '/sendMessage/',
      data : JSON.stringify ({ sent_text : $('#messageToBot').val()}),
      dataType: 'json',
      contentType: 'application/json;charset=UTF-8',
      success: function (data) {
        if (data.answer) {
          $('#messageToBot').val('');
          $('#botSays').val(data.answer);
           setColorByMood(data.mood)
        }
      },
      error: function (xhr, ajaxOptions, thrownError) {
        $('#botSays').val('Your message could not be sent.');
      }
    });
  }
}

firstColor = "#23074d"
secondColor = "#cc5333"

function lighten(color, luminosity) {
	// validate hex string
	color = new String(color).replace(/[^0-9a-f]/gi, '');
	if (color.length < 6) {
		color = color[0]+ color[0]+ color[1]+ color[1]+ color[2]+ color[2];
	}
	luminosity = luminosity || 0;

	// convert to decimal and change luminosity
	var newColor = "#", c, i, black = 0, white = 255;
	for (i = 0; i < 3; i++) {
		c = parseInt(color.substr(i*2,2), 16);
		c = Math.round(Math.min(Math.max(black, c + (luminosity * white)), white)).toString(16);
		newColor += ("00"+c).substr(c.length);
	}
	return newColor; 
}


function setColorByMood(myMood){
  if (mood != null) {
      c1 = lighten(firstColor, 0.08 * myMood);
      c2 = lighten(secondColor, 0.08 * myMood);
      setBackgroundColor("bottom right", c2, c1);
  }
}


function setBackgroundColor(orientation,colorOne, colorTwo){
  var dom = document.getElementById('bg');
  var orientation = "bottom right";
  dom.style.backgroundImage =  getCssValuePrefix() + 'linear-gradient('+ orientation + ', ' + colorOne + ', ' + colorTwo + ')';
}

function getCssValuePrefix()
{
    var rtrnVal = '';//default to standard syntax
    var prefixes = ['-o-', '-ms-', '-moz-', '-webkit-'];
    // Create a temporary DOM object for testing
    var dom = document.createElement('div');
    for (var i = 0; i < prefixes.length; i++)
    {
        // Attempt to set the style
        dom.style.background = prefixes[i] + 'linear-gradient(#000000, #ffffff)';
        // Detect if the style was successfully set
        if (dom.style.background)
        {
            rtrnVal = prefixes[i];
        }
    }

    dom = null;
    delete dom;

    return rtrnVal;
  }


