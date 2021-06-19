function formValidation()
{
var uemail = document.registration.email;
var passid = document.registration.passid;
var umobile = document.registration.mobile;
var fname = document.registration.fname;
var lname = document.registration.lname;
var nname = document.registration.nname;
if(ValidateEmail(uemail))
{
if(passid_validation(passid,7,18))
{
if(mobile_validation(umobile))
{
if(allLetter(fname))
{
if(allLetter(lname))
{
if(allLetter(nname))
{
}
}
}
} 
}
}
return true;
}

function ValidateEmail(uemail)
{
var uemail_len = uemail.value.length;
if(uemail_len == 0)
{
  alert("Enter email address");
  uemail.focus();
  return false;
} else {
var mailformat = /^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/;
if(uemail.value.match(mailformat))
{
return true;
}
else
{
alert("You have entered an invalid email address!");
uemail.focus();
return false;
}
}
}
function passid_validation(passid,mx,my)
{
var passid_len = passid.value.length;
if (passid_len == 0 ||passid_len >= my || passid_len < mx)
{
alert("Password should not be empty / length be between "+mx+" to "+my);
passid.focus();
return false;
}
return true;
}
  function mobile_validation(umobile)
  {
  var phoneno = /^\d{10}$/;
  if(umobile.value.match(phoneno))
        {
      return true;
        }
      else
        {
        alert("Phone number needs to be 10 digit");
        umobile.focus();
        return false;
        }
  }
function allLetter(uname)
{ 
var uname_len = uname.value.length;
var letters = /^[A-Za-z]+$/;
if(uname_len == 0)
 {
  alert("Please enter name");
  return false;
 } else {
  if(uname.value.match(letters))
  {
  return true;
  }
  else
  {
  alert('Username must have alphabet characters only');
  uname.focus();
  return false;
}
}
}
