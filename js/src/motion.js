$(document).ready(function(){NexT.motion={};var t={lines:[],push:function(i){this.lines.push(i)},init:function(){this.lines.forEach(function(i){i.init()})},arrow:function(){this.lines.forEach(function(i){i.arrow()})},close:function(){this.lines.forEach(function(i){i.close()})}};function i(i){this.el=$(i.el),this.status=$.extend({},{init:{width:"100%",opacity:1,left:0,rotateZ:0,top:0}},i.status)}i.prototype.init=function(){this.transform("init")},i.prototype.arrow=function(){this.transform("arrow")},i.prototype.close=function(){this.transform("close")},i.prototype.transform=function(i){this.el.velocity("stop").velocity(this.status[i])};var e=new i({el:".sidebar-toggle-line-first",status:{arrow:{width:"50%",rotateZ:"-45deg",top:"2px"},close:{width:"100%",rotateZ:"-45deg",top:"5px"}}}),o=new i({el:".sidebar-toggle-line-middle",status:{arrow:{width:"90%"},close:{opacity:0}}}),n=new i({el:".sidebar-toggle-line-last",status:{arrow:{width:"50%",rotateZ:"45deg",top:"-2px"},close:{width:"100%",rotateZ:"45deg",top:"-5px"}}});t.push(e),t.push(o),t.push(n);var s,r,a="320px";({toggleEl:$(".sidebar-toggle"),dimmerEl:$("#sidebar-dimmer"),sidebarEl:$(".sidebar"),isSidebarVisible:!1,init:function(){this.toggleEl.on("click",this.clickHandler.bind(this)),this.dimmerEl.on("click",this.clickHandler.bind(this)),this.toggleEl.on("mouseenter",this.mouseEnterHandler.bind(this)),this.toggleEl.on("mouseleave",this.mouseLeaveHandler.bind(this)),this.sidebarEl.on("touchstart",this.touchstartHandler.bind(this)),this.sidebarEl.on("touchend",this.touchendHandler.bind(this)),this.sidebarEl.on("touchmove",function(i){i.preventDefault()}),$(document).on("sidebar.isShowing",function(){NexT.utils.isDesktop()&&$("body").velocity("stop").velocity({paddingRight:a},200)}).on("sidebar.isHiding",function(){})},clickHandler:function(){this.isSidebarVisible?this.hideSidebar():this.showSidebar(),this.isSidebarVisible=!this.isSidebarVisible},mouseEnterHandler:function(){this.isSidebarVisible||t.arrow()},mouseLeaveHandler:function(){this.isSidebarVisible||t.init()},touchstartHandler:function(i){s=i.originalEvent.touches[0].clientX,r=i.originalEvent.touches[0].clientY},touchendHandler:function(i){var t=i.originalEvent.changedTouches[0].clientX,e=i.originalEvent.changedTouches[0].clientY;30<t-s&&Math.abs(e-r)<20&&this.clickHandler()},showSidebar:function(){var i=this;t.close(),this.sidebarEl.velocity("stop").velocity({width:a},{display:"block",duration:200,begin:function(){$(".sidebar .motion-element").velocity("transition.slideRightIn",{stagger:50,drag:!0,complete:function(){i.sidebarEl.trigger("sidebar.motion.complete")}})},complete:function(){i.sidebarEl.addClass("sidebar-active"),i.sidebarEl.trigger("sidebar.didShow")}}),this.sidebarEl.trigger("sidebar.isShowing")},hideSidebar:function(){NexT.utils.isDesktop()&&$("body").velocity("stop").velocity({paddingRight:0}),this.sidebarEl.find(".motion-element").velocity("stop").css("display","none"),this.sidebarEl.velocity("stop").velocity({width:0},{display:"none"}),t.init(),this.sidebarEl.removeClass("sidebar-active"),this.sidebarEl.trigger("sidebar.isHiding"),$(".post-toc-wrap")||("block"===$(".site-overview-wrap").css("display")?$(".post-toc-wrap").removeClass("motion-element"):$(".post-toc-wrap").addClass("motion-element"))}}).init(),NexT.motion.integrator={queue:[],cursor:-1,add:function(i){return this.queue.push(i),this},next:function(){this.cursor++;var i=this.queue[this.cursor];$.isFunction(i)&&i(NexT.motion.integrator)},bootstrap:function(){this.next()}},NexT.motion.middleWares={logo:function(i){var t=[],e=$(".brand"),o=$(".site-title"),n=$(".site-subtitle"),s=$(".logo-line-before i"),r=$(".logo-line-after i");function a(i){return(i=Array.isArray(i)?i:[i]).every(function(i){return 0<i.length})}function l(i,t){return{e:$(i),p:{translateX:t},o:{duration:500,sequenceQueue:!1}}}0<e.length&&t.push({e:e,p:{opacity:1},o:{duration:200}}),NexT.utils.isMist()&&a([s,r])&&t.push(l(s,"100%"),l(r,"-100%")),a(o)&&t.push({e:o,p:{opacity:1,top:0},o:{duration:200}}),a(n)&&t.push({e:n,p:{opacity:1,top:0},o:{duration:200}}),CONFIG.motion.async&&i.next(),0<t.length?(t[t.length-1].o.complete=function(){i.next()},$.Velocity.RunSequence(t)):i.next()},menu:function(i){CONFIG.motion.async&&i.next(),$(".menu-item").velocity("transition.slideDownIn",{display:null,duration:200,complete:function(){i.next()}})},postList:function(i){var t,e=$(".post-block, .pagination, .comments"),o=CONFIG.motion.transition.post_block,n=$(".post-header"),s=CONFIG.motion.transition.post_header,r=$(".post-body"),a=CONFIG.motion.transition.post_body,l=$(".collection-title, .archive-year"),c=CONFIG.motion.transition.coll_header,d=$(".sidebar-inner"),u=CONFIG.motion.transition.sidebar;0<e.length?((t=window.postMotionOptions||{stagger:100,drag:!0}).complete=function(){CONFIG.motion.transition.sidebar&&(NexT.utils.isPisces()||NexT.utils.isGemini())&&d.css({transform:"initial"}),i.next()},CONFIG.motion.transition.post_block&&e.velocity("transition."+o,t),CONFIG.motion.transition.post_header&&n.velocity("transition."+s,t),CONFIG.motion.transition.post_body&&r.velocity("transition."+a,t),CONFIG.motion.transition.coll_header&&l.velocity("transition."+c,t),CONFIG.motion.transition.sidebar&&(NexT.utils.isPisces()||NexT.utils.isGemini())&&d.velocity("transition."+u,t)):i.next(),CONFIG.motion.async&&i.next()},sidebar:function(i){"always"===CONFIG.sidebar.display&&NexT.utils.displaySidebar(),i.next()}}});