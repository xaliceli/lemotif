function expand(element) {
    var x = document.getElementById(element);
    if (x.style.display === "none") {
        x.style.display = "inline-block";
    } else {
        x.style.display = "none";
    }
}

$(function () {
    var emotions = Object.keys(emotionsDict);
    var subjects = subjectsList;

    function split(val) {
        return val.split(/,\s*/);
    }

    function extractLast(term) {
        return split(term).pop();
    }

    for (i = 0; i < 4; i++) {
        subjId = '#subjects' + (i + 1);
        emotId = '#emotions' + (i + 1);
        $(subjId)
            .autocomplete({
                source: subjects
            })
        .autocomplete("instance")._renderItem = function (ul, item) {
            return $("<li>")
                .append("<div> <img src='" + img_url + item.label + ".png'>"
                    + "<div class='ui-menu-item-label'>" + item.label + "</div></div>")
                .appendTo(ul);
        };
        $(emotId)
            .on("keydown", function (event) {
                if (event.keyCode === $.ui.keyCode.TAB &&
                    $(this).autocomplete("instance").menu.active) {
                    event.preventDefault();
                }
            })
            .autocomplete({
                minLength: 0,
                source: function (request, response) {
                    // delegate back to autocomplete, but extract the last term
                    response($.ui.autocomplete.filter(
                        emotions, extractLast(request.term)));
                },
                focus: function () {
                    // prevent value inserted on focus
                    return false;
                },
                select: function (event, ui) {
                    var terms = split(this.value);
                    // remove the current input
                    terms.pop();
                    // add the selected item
                    terms.push(ui.item.value);
                    // add placeholder to get the comma-and-space at the end
                    terms.push("");
                    this.value = terms.join(", ");
                    return false;
                }
            })
            .autocomplete("instance")._renderItem = function (ul, item) {
            return $("<li>")
                .append("<div style='background-color:" + emotionsDict[item.value]['hex'] + "; color: #fff'>"
                    + item.label + "</div>")
                .appendTo(ul);
        };
    }
});