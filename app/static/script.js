String.prototype.isEmpty = function() {
    return (this.length === 0 || !this.trim());
};

function expand(element) {
    var x = document.getElementById(element);
    if (x.style.display === "none" || x.style.display === "") {
        x.style.display = "inline-block";
    } else {
        x.style.display = "none";
    }
}

function expand_and_hide(element, form) {
    var x = document.getElementById(element);
    if (x.style.display === "none" || x.style.display === "") {
        x.style.display = "inline-block";
    } else {
        x.style.display = "none";
    }
    var settings = ['settings-carpet', 'settings-circle', 'settings-glass',
        'settings-tile', 'settings-string', 'settings-watercolors'];
    settings = settings.filter(s => s !== element);
    for (var i = 0; i < settings.length; i++) {
        document.getElementById(settings[i]).style.display = "none";
    }

    var f = document.getElementById(form);
    var t1 = document.getElementById('text1').value;
    var t2 = document.getElementById('text2').value;
    var t3 = document.getElementById('text3').value;

    console.log(t1.isEmpty());
    console.log((!t1.isEmpty() || !t2.isEmpty() || !t3.isEmpty()));

    if ((!t1.isEmpty() || !t2.isEmpty() || !t3.isEmpty())) {
        f.submit();
        document.getElementById('vis').innerHTML = 'Please wait, generating...';
    }

}

function expand_and_toggle(element, toggle) {
    var x = document.getElementById(element);
    var t = document.getElementById(toggle);

    if (x.style.display === "none" || x.style.display === "") {
        x.style.display = "inline-block";
        t.value = "inline-block";
    } else {
        x.style.display = "none";
        t.value = "none";
    }
}

// $('#download').click(function(){
//      $(this).parent().attr('href', document.getElementById('canvas').toDataURL());
//      $(this).parent().attr('download', "myPicture.png");
// });
//
// html2canvas($('#motifs').get(0)).then(function(canvas) {
//     var lemotif = canvas.toDataURL("image/png");
// });

// $(function () {
//     var emotions = Object.keys(emotionsDict);
//     var subjects = subjectsList;
//
//     function split(val) {
//         return val.split(/,\s*/);
//     }
//
//     function extractLast(term) {
//         return split(term).pop();
//     }
//
//     for (i = 0; i < 4; i++) {
//         subjId = '#subjects' + (i + 1);
//         emotId = '#emotions' + (i + 1);
//         $(subjId)
//             .autocomplete({
//                 source: subjects,
//                 minLength: 0,
//                 change: function (event, ui) {
//                     if (!ui.item) {
//                         $(this).val('');
//                         $(this).attr("placeholder", 'Invalid input, please try again.');
//                     }
//                 }
//             })
//             .click(function () {
//                 $(this).autocomplete("search", "");
//             })
//             .autocomplete("instance")._renderItem = function (ul, item) {
//             return $("<li>")
//                 .append("<div> <img src='" + img_url + item.label + ".png'>"
//                     + "<div class='ui-menu-item-label'>" + item.label + "</div></div>")
//                 .appendTo(ul);
//         };
//         $(emotId)
//             .on("keydown", function (event) {
//                 if (event.keyCode === $.ui.keyCode.TAB &&
//                     $(this).autocomplete("instance").menu.active) {
//                     event.preventDefault();
//                 }
//             })
//             .autocomplete({
//                 minLength: 0,
//                 source: function (request, response) {
//                     // delegate back to autocomplete, but extract the last term
//                     response($.ui.autocomplete.filter(
//                         emotions, extractLast(request.term)));
//                 },
//                 focus: function () {
//                     // prevent value inserted on focus
//                     return false;
//                 },
//                 select: function (event, ui) {
//                     var terms = split(this.value);
//                     // remove the current input
//                     terms.pop();
//                     // add the selected item
//                     terms.push(ui.item.value);
//                     // add placeholder to get the comma-and-space at the end
//                     terms.push("");
//                     this.value = terms.join(", ");
//                     return false;
//                 },
//                 change: function (event, ui) {
//                     if (!ui.item) {
//                         $(this).val('');
//                         $(this).attr("placeholder", 'Invalid input, please try again.');
//                     }
//                 }
//             })
//             .click(function () {
//                 $(this).autocomplete("search", "");
//             })
//             .autocomplete("instance")._renderItem = function (ul, item) {
//             return $("<li>")
//                 .append("<span style='color:" + emotionsDict[item.value]['hex'] + ";'>" + "&#9632;" + "</span>"
//                     + item.label)
//                 .appendTo(ul);
//         };
//     }
// });