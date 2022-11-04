$(document).ready(function () {
    $(document).bind("ajaxSend", function(){
        $("#loading").show();
    }).bind("ajaxComplete", function(){
        $("#loading").hide();
    });
})