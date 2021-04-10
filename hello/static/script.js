	function returnWordsearch(word, callback){
        $.ajax({
          type: "GET",
          url: "/returnWordsearch/",  
          data: { 
            csrfmiddlewaretoken: '{{ csrf_token }}',
            word: word},
          success: callback,
          complete: function(data){}
        });
    } 

    function getWordsearch(word){
	    returnWordsearch(word, function(response) {
	    	var imurls = response[0];
	    	var imcols = response[1];
	    	count = imurls.length;

	    	if (count == 0) {
                document.getElementById('wordIms').innerHTML= "No Results"
                return
            }
            
            document.getElementById('wordIms').innerHTML= ""

            var container = document.getElementById('wordIms');

            var tot = 20; 
            if (count < tot) tot = count;

            for (let i = 0; i < tot; i++) {
            	var img = document.createElement('img');
            	img.src = imurls[i];
                img.className= "display-im"

                cols = imcols[i]

                var elem = document.createElement('div');
                elem.setAttribute("class", "im-and-pal");


                var pal = document.createElement('div');
                pal.setAttribute("class", "palette");

                var col1 = document.createElement('div');
                col1.setAttribute("class", "pal-color");
                var hex1 = 'rgb(' + cols[0][0]+ ',' + cols[0][1] + ',' + cols[0][2] + ')'
                col1.style.cssText = 'background-color:' + hex1 + ';'

                var col2 = document.createElement('div');
                col2.setAttribute("class", "pal-color"); 
                var hex2 = 'rgb(' + cols[1][0]+ ',' + cols[1][1] + ',' + cols[1][2] + ')'
                col2.style.cssText = 'background-color:' + hex2 + ';'

            
                var col3 = document.createElement('div');
                col3.setAttribute("class", "pal-color");
                var hex3 = 'rgb(' + cols[2][0]+ ',' + cols[2][1] + ',' + cols[2][2]+ ')'
                col3.style.cssText = 'background-color:' + hex3 + ';'

                var col4 = document.createElement('div');
                col4.setAttribute("class", "pal-color");
                var hex4 = 'rgb(' + cols[3][0]+ ',' + cols[3][1] + ',' + cols[3][2]+ ')'
                col4.style.cssText = 'background-color:' + hex4 + ';'

                var col5 = document.createElement('div');
                col5.setAttribute("class", "pal-color");
                var hex5 = 'rgb(' + cols[4][0]+ ',' + cols[4][1] + ',' + cols[4][2] +')'
                col5.style.cssText = 'background-color:' + hex5 + ';'

                pal.appendChild(col1);
                pal.appendChild(col2);
                pal.appendChild(col3);
                pal.appendChild(col4);
                pal.appendChild(col5);
                
                elem.append(img);
                elem.append(pal);
              
                container.appendChild(elem);

            }

	    });
    }


    function returnAll(img, callback){
        $.ajax({
          type: "GET",
          url: "/returnAll/",  
          data: { 
            csrfmiddlewaretoken: '{{ csrf_token }}',
            imsrc: img},
          success: callback,
          complete: function(data){}
        });
    } 


    function getims(){
    	var word = document.getElementById("wordbox").value
    	getWordsearch(word)
    }


    function getAll(imgsrc){
	    returnAll(imgsrc, function(response) {
	    	var palette = response[0];

	    	var binim = response[1];
	    	var pix = response[2];

	    	var lowsat = response[3];
	    	var highsat = response[4];
	    	var valu = response[5];
	    	var compl = response[6];
	    	var compl2 = response[7];
	    	var compl3 = response[8];

	    	var edge = response[9];
	    	var switch1 = response[10];
		    var switch2 = response[11];
		    var switch3 = response[12];

		    var imurls = response[13];
		    var imcols = response[14];

            var labels = response[15];
            // CONSOLE.LOG
            document.getElementById("labels").innerHTML = labels


	    	fillPalette(palette, 'a');
	    	fillPalette(lowsat, 'b');
	    	fillPalette(highsat, 'c');
	    	fillPalette(valu, 'd');
	    	fillPalette(compl, 'e');
	    	fillPalette(compl2, 'f');
	    	fillPalette(compl3, 'g');
	    	

	    	document.getElementById("im1").src = binim
	    	document.getElementById("im2").src = pix
	    	document.getElementById("im3").src = edge

	    	document.getElementById("im4").src = switch1
	    	document.getElementById("im5").src = switch2
	    	document.getElementById("im6").src = switch3




	    	document.getElementById('palIms').innerHTML= ""
            var container = document.getElementById('palIms');
            for (let i = 0; i < 10; i++) {
            	var img = document.createElement('img');
            	img.src = imurls[i];
                img.className= "display-im"

                cols = imcols[i]

                var elem = document.createElement('div');
                elem.setAttribute("class", "im-and-pal");


                var pal = document.createElement('div');
                pal.setAttribute("class", "palette");

                var col1 = document.createElement('div');
                col1.setAttribute("class", "pal-color");
                var hex1 = 'rgb(' + cols[0][0]+ ',' + cols[0][1] + ',' + cols[0][2] + ')'
                col1.style.cssText = 'background-color:' + hex1 + ';'

                var col2 = document.createElement('div');
                col2.setAttribute("class", "pal-color"); 
                var hex2 = 'rgb(' + cols[1][0]+ ',' + cols[1][1] + ',' + cols[1][2] + ')'
                col2.style.cssText = 'background-color:' + hex2 + ';'

            
                var col3 = document.createElement('div');
                col3.setAttribute("class", "pal-color");
                var hex3 = 'rgb(' + cols[2][0]+ ',' + cols[2][1] + ',' + cols[2][2]+ ')'
                col3.style.cssText = 'background-color:' + hex3 + ';'

                var col4 = document.createElement('div');
                col4.setAttribute("class", "pal-color");
                var hex4 = 'rgb(' + cols[3][0]+ ',' + cols[3][1] + ',' + cols[3][2]+ ')'
                col4.style.cssText = 'background-color:' + hex4 + ';'

                var col5 = document.createElement('div');
                col5.setAttribute("class", "pal-color");
                var hex5 = 'rgb(' + cols[4][0]+ ',' + cols[4][1] + ',' + cols[4][2] +')'
                col5.style.cssText = 'background-color:' + hex5 + ';'

                pal.appendChild(col1);
                pal.appendChild(col2);
                pal.appendChild(col3);
                pal.appendChild(col4);
                pal.appendChild(col5);
                
                elem.append(img);
                elem.append(pal);
              
                container.appendChild(elem);

            }



	    });
    }

    function fillPalette(palette, id) {
    	var hex1 = 'rgb(' + palette[0][0]+ ',' + palette[0][1] + ',' + palette[0][2] + ')';
    	var hex2 = 'rgb(' + palette[1][0]+ ',' + palette[1][1] + ',' + palette[1][2] + ')';
    	var hex3 = 'rgb(' + palette[2][0]+ ',' + palette[2][1] + ',' + palette[2][2] + ')';
    	var hex4 = 'rgb(' + palette[3][0]+ ',' + palette[3][1] + ',' + palette[3][2] + ')';
    	var hex5 = 'rgb(' + palette[4][0]+ ',' + palette[4][1] + ',' + palette[4][2] + ')';

    	var col1 = document.getElementById(id + '1');
    	var col2 = document.getElementById(id + '2');
    	var col3 = document.getElementById(id + '3');
    	var col4 = document.getElementById(id + '4');
    	var col5 = document.getElementById(id + '5');

    	col1.style.backgroundColor = hex1;
    	col2.style.backgroundColor = hex2;
    	col3.style.backgroundColor = hex3;
    	col4.style.backgroundColor = hex4;
    	col5.style.backgroundColor = hex5;

    }