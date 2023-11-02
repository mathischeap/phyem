
.. _message-form:

Message form
============

📨 To send Yi a message, please use the form below.

.. raw:: html

    <form name="contact" method="POST" data-netlify="true">
      <p>
        <label>Your Name: <input type="text" name="name" /></label>
      </p>
      <p>
        <label>Your Email: <input type="email" name="email" /></label>
      </p>
      <p>
        <label>Your Role: <select name="role[]" multiple>
          <option value="leader">Leader</option>
          <option value="follower">Follower</option>
        </select></label>
      </p>
      <p>
        <label>Message: <textarea name="message"></textarea></label>
      </p>
      <p>
        <button type="submit">Send</button>
      </p>
    </form>



| 🚨 -You can upload only one file. Please merge your files. Sorry for the inconvenience.
| 🚨 -The file has a maximum size limit of ~8 MB.
| 🚨 -File uploading times out after 30 seconds.

🔗 Message submission service provided by *netlify*. All messages will be filtered for spam using Akismet.
