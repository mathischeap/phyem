
.. _message-form:

Message form
============

📨 To send Yi a message, please use the form below.

.. raw:: html

   <form name="Message Forms" enctype="multipart/form-data" data-netlify="true">
     <input type="hidden" name="subject"
     value="Message received by [phyem.org]" />
     <p>
       <label>Your name: <input size="20" type="text" name="name" /></label>
     </p>
     <p>
       <label>Your email: <input size="20" type="email" name="email" /></label>
     </p>
     <p>
       <label>Title: <input size="45" type="title" name="title" /></label>
     </p>
     <p>
       <label>Message: <textarea rows="10" cols="90" name="message"></textarea></label>
     </p>
     <p>
        <label>
          <span>Attach file:</span>
          <input name="file" type="file"/>
        </label>
     </p>
     <button>Submit</button>
    </form>
    <p class="result"></p>

| 🚨 -You can upload only one file. Please merge your files. Sorry for the inconvenience.
| 🚨 -The file has a maximum size limit of ~8 MB.
| 🚨 -File uploading times out after 30 seconds.

🔗 Message submission service provided by *netlify*. All messages will be filtered for spam using Akismet.
