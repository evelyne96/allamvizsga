You can find the current version at:
 https://calm-atoll-55967.herokuapp.com/bot
 
  final FragmentManager fragmentManager = getSupportFragmentManager();
        final FireStoreProvider fp = new FireStoreProvider();
        fp.getData(DBUtils.REQUEST_COLLECTION, new OnCompleteListener<QuerySnapshot>() {
            @Override
            public void onComplete(@NonNull Task<QuerySnapshot> task) {
                final List<Request> requests = task.getResult().toObjects(Request.class);
                for (int i = 0;i<requests.size();i++){
                    final int x = i;
                    Query q = fp.createQueryOn(DBUtils.MANAGER_REQUESTS,"==", "requestId",
                                    task.getResult().getDocuments().get(i).getId());
                    fp.getQueryResult(q, new OnCompleteListener<QuerySnapshot>() {
                        @Override
                        public void onComplete(@NonNull Task<QuerySnapshot> task) {
                            final List<ManagerRequest> mr = task.getResult().toObjects(ManagerRequest.class);
                            for (int j = 0;j<mr.size();j++) {
                                fp.update(DBUtils.MANAGER_REQUESTS, task.getResult().getDocuments().get(j).getId(),
                                        "date", requests.get(x).getStartDate(), null, null);
                            }
                        }
                    }, null);
                }
            }
        }, new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception e) {

            }
        });